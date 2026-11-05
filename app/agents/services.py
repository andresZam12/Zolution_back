"""
Agent Config service layer — business logic and prompt engineering engine.

The central service of the Zolution platform. It takes a business owner's
onboarding answers and assembles a production-ready Master System Prompt
following the structure from Section 6 of the project documentation.

The prompt engineering engine (`_build_system_prompt`) is the primary value
differentiator of the product. It is kept here, in the service layer, not
in a route or a script, so it can be:
- Unit tested without HTTP overhead
- Evolved without touching the API contract
- Called from background jobs if needed in the future
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models import AgentConfig
from app.agents.schemas import (
    AgentConfigUpdate,
    AgentStatusResponse,
    AIEnrichRequest,
    AIEnrichResponse,
    OnboardingAnswers,
    ServiceItem,
)
from app.core.database import bind_tenant
from app.llm_providers.base import ConversationMessage as LLMConversationMessage
from app.llm_providers.factory import LLMProviderFactory
from app.tenants.models import Organization

logger = logging.getLogger(__name__)


class AgentConfigService:
    """
    Business logic for AgentConfig management and prompt generation.
    """

    async def get_for_org(
        self,
        organization_id: UUID,
        db: AsyncSession,
    ) -> AgentConfig:
        """
        Fetch the agent config for an organization.
        Auto-provisions one if the organization exists in DB but lacks a config row.

        Raises:
            HTTPException 404: No agent config found for the organization.
        """
        await bind_tenant(db, organization_id)

        result = await db.execute(
            select(AgentConfig).where(AgentConfig.organization_id == organization_id)
        )
        config = result.scalar_one_or_none()
        if config is None:
            org_res = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            org = org_res.scalar_one_or_none()
            if org is not None:
                config = AgentConfig(
                    organization_id=organization_id,
                    status="draft",
                    llm_provider="openai",
                    llm_model="gpt-4o-mini",
                    onboarding_answers={
                        "companyName": org.name,
                        "business_name": org.name,
                    },
                )
                db.add(config)
                await db.flush()
                return config

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No agent config found for organization {organization_id}.",
            )
        return config

    async def get_status(
        self,
        organization_id: UUID,
        db: AsyncSession,
    ) -> AgentStatusResponse:
        """
        Get the lightweight status for the dashboard semaphore widget.

        Returns a 3-state view:
            active    → Agent is live and handling conversations
            draft     → In onboarding/configuration
            suspended → Disabled (payment issue, policy violation, etc.)
        """
        config = await self.get_for_org(organization_id, db)
        answers = config.onboarding_answers or {}
        onboarding_complete = _is_onboarding_complete(answers)

        return AgentStatusResponse(
            status=config.status,
            has_system_prompt=bool(config.system_prompt_generated),
            onboarding_complete=onboarding_complete,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
        )

    async def update_onboarding_answers(
        self,
        organization_id: UUID,
        answers: OnboardingAnswers,
        db: AsyncSession,
    ) -> AgentConfig:
        """
        Replace or merge the onboarding answers document.

        Persists as JSONB and synchronizes Organization.name if companyName or
        business_name is provided. If system_prompt_generated is included in the
        answers payload, also stores it and marks the agent active.

        Raises:
            HTTPException 404: No agent config for this org.
        """
        config = await self.get_for_org(organization_id, db)
        dumped = answers.model_dump(exclude_none=False)

        # Normalize aliases between frontend camelCase and backend snake_case
        company_name = dumped.get("business_name") or dumped.get("companyName")
        industry = dumped.get("business_type") or dumped.get("industry")
        agent_name = dumped.get("agent_name") or dumped.get("agentName")

        if company_name:
            dumped["business_name"] = company_name
            dumped["companyName"] = company_name
        if industry:
            dumped["business_type"] = industry
            dumped["industry"] = industry
        if agent_name:
            dumped["agent_name"] = agent_name
            dumped["agentName"] = agent_name

        existing_answers = dict(config.onboarding_answers or {})
        existing_answers.update(
            {
                k: v
                for k, v in dumped.items()
                if v is not None
                or k in ("business_name", "business_type", "services", "hours", "agent_name")
            }
        )
        config.onboarding_answers = existing_answers

        if dumped.get("system_prompt_generated"):
            config.system_prompt_generated = dumped["system_prompt_generated"]
            config.status = "active"

        # Also synchronize Organization.name in PostgreSQL when company_name is provided
        if company_name and len(company_name.strip()) >= 2:
            org_res = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            org = org_res.scalar_one_or_none()
            if org:
                org.name = company_name.strip()
                db.add(org)

        db.add(config)
        logger.info("Onboarding answers updated for org %s", organization_id)
        return config

    async def update_config(
        self,
        organization_id: UUID,
        data: AgentConfigUpdate,
        db: AsyncSession,
    ) -> AgentConfig:
        """
        Partially update the AgentConfig (system prompt, LLM provider/model, status,
        onboarding_answers metadata) and optionally Organization.name.
        """
        config = await self.get_for_org(organization_id, db)

        if data.system_prompt_generated is not None:
            config.system_prompt_generated = data.system_prompt_generated
        if data.llm_provider is not None:
            from app.llm_providers.factory import SUPPORTED_PROVIDERS

            if data.llm_provider not in SUPPORTED_PROVIDERS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Provider '{data.llm_provider}' is not supported. "
                    f"Supported: {sorted(SUPPORTED_PROVIDERS)}",
                )
            config.llm_provider = data.llm_provider
        if data.llm_model is not None:
            config.llm_model = data.llm_model
        if data.status is not None and data.status in ("active", "draft", "suspended"):
            config.status = data.status

        if data.onboarding_answers is not None:
            merged = dict(config.onboarding_answers or {})
            merged.update(data.onboarding_answers)
            if merged.get("companyName") and not merged.get("business_name"):
                merged["business_name"] = merged["companyName"]
            if merged.get("industry") and not merged.get("business_type"):
                merged["business_type"] = merged["industry"]
            config.onboarding_answers = merged
            from sqlalchemy.orm.attributes import flag_modified

            flag_modified(config, "onboarding_answers")

        new_org_name = data.company_name or (
            data.onboarding_answers.get("companyName") if data.onboarding_answers else None
        )
        if new_org_name and len(new_org_name.strip()) >= 2:
            org_res = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            org = org_res.scalar_one_or_none()
            if org:
                org.name = new_org_name.strip()
                db.add(org)

        db.add(config)
        await db.flush()
        await db.refresh(config)  # Load server-generated timestamps for the response
        logger.info("AgentConfig updated for org %s", organization_id)
        return config

    async def generate_system_prompt(
        self,
        organization_id: UUID,
        db: AsyncSession,
    ) -> AgentConfig:
        """
        Build the Master System Prompt from the stored onboarding answers.

        This is the core value-add of the platform. The generated prompt
        is stored in `system_prompt_generated` and used in every LLM call
        for this tenant's agent.

        Raises:
            HTTPException 400: Onboarding answers are missing or incomplete.
            HTTPException 404: No agent config found.
        """
        config = await self.get_for_org(organization_id, db)

        if not config.onboarding_answers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot generate prompt: onboarding answers are not set. "
                "Complete the onboarding form first.",
            )

        answers = OnboardingAnswers.model_validate(config.onboarding_answers)

        if not _is_onboarding_complete(config.onboarding_answers):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot generate prompt: onboarding is incomplete. "
                "Required fields: business_name, business_type, services (at least 1), "
                "hours (at least 1 day), agent_name.",
            )

        prompt = _build_system_prompt(answers)
        config.system_prompt_generated = prompt
        db.add(config)
        logger.info(
            "System prompt generated for org %s (%d chars)",
            organization_id,
            len(prompt),
        )
        return config

    async def activate(
        self,
        organization_id: UUID,
        db: AsyncSession,
    ) -> AgentConfig:
        """
        Activate the agent (set status = 'active').

        Validates that a system prompt has been generated before allowing
        activation, preventing misconfigured agents from going live.

        Raises:
            HTTPException 400: No system prompt generated yet.
            HTTPException 404: No agent config found.
        """
        config = await self.get_for_org(organization_id, db)

        if not config.system_prompt_generated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot activate agent: system prompt has not been generated yet. "
                "Call POST /agents/me/generate-prompt first.",
            )

        config.status = "active"
        db.add(config)
        logger.info("Agent activated for org %s", organization_id)
        return config

    async def set_llm_provider(
        self,
        organization_id: UUID,
        provider: str,
        model: str,
        db: AsyncSession,
    ) -> AgentConfig:
        """
        Update the LLM provider and model for this tenant's agent.

        Called after the diagnostic benchmark selects the best provider.

        Raises:
            HTTPException 400: Provider not supported.
        """
        from app.llm_providers.factory import SUPPORTED_PROVIDERS

        if provider not in SUPPORTED_PROVIDERS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider}' is not supported. "
                f"Supported: {sorted(SUPPORTED_PROVIDERS)}",
            )

        config = await self.get_for_org(organization_id, db)
        config.llm_provider = provider
        config.llm_model = model
        db.add(config)
        logger.info("LLM provider updated for org %s: %s/%s", organization_id, provider, model)
        return config

    def _get_fallback_enrichment(self, req: AIEnrichRequest) -> dict:
        """Returns realistic curated profile by industry if LLM is unavailable."""
        ind = f"{req.industry} {req.business_name}".lower()

        if any(k in ind for k in ("dent", "odont", "salud", "médic", "clinic", "sonris")):
            return {
                "agent_name": "Sofía",
                "tone": "Cálido, empático y profesional",
                "services": [
                    {
                        "name": "Valoración Odontológica y Diagnóstico",
                        "price_cop": 30000,
                        "duration_minutes": 30,
                        "description": "Examen clínico completo y plan de tratamiento",
                    },
                    {
                        "name": "Limpieza Dental Profunda (Profilaxis)",
                        "price_cop": 80000,
                        "duration_minutes": 45,
                        "description": "Eliminación de sarro, placa y pulido dental",
                    },
                    {
                        "name": "Control o Ajuste de Ortodoncia",
                        "price_cop": 55000,
                        "duration_minutes": 30,
                        "description": "Revisión mensual de brackets y cambio de arcos",
                    },
                    {
                        "name": "Calza o Restauración en Resina",
                        "price_cop": 90000,
                        "duration_minutes": 45,
                        "description": "Tratamiento estético de caries por pieza dental",
                    },
                    {
                        "name": "Blanqueamiento Dental LED",
                        "price_cop": 280000,
                        "duration_minutes": 60,
                        "description": "Aclaramiento dental seguro en consultorio",
                    },
                ],
                "schedule_weekdays": "Lunes a Viernes: 8:00 AM - 6:00 PM",
                "schedule_saturdays": "Sábados: 8:00 AM - 1:00 PM",
                "holiday_policy": "Festivos: Solo urgencias o cerrado",
                "cancellation_policy": "Avisar cancelación o cambio con mínimo 2 horas de anticipación",
                "tolerance_minutes": 10,
                "payment_methods": [
                    "Nequi / Daviplata",
                    "Transferencia Bancolombia",
                    "Tarjetas débito/crédito",
                    "Efectivo",
                ],
                "faqs": [
                    {
                        "question": "¿Aceptan seguros o medicina prepagada?",
                        "answer": "Atendemos de forma particular y expedimos facturas para reembolso según tu plan de medicina prepagada.",
                    },
                    {
                        "question": "¿Qué pasa si llego tarde?",
                        "answer": "Tenemos una tolerancia de 10 minutos; pasado ese tiempo reprogramamos tu cita para no afectar la atención del siguiente paciente.",
                    },
                ],
            }
        elif any(
            k in ind for k in ("spa", "belleza", "barber", "peluquer", "estétic", "uñas", "masaje")
        ):
            return {
                "agent_name": "Camila",
                "tone": "Amable, relajante, atento y cordial",
                "services": [
                    {
                        "name": "Corte de Cabello y Estilismo",
                        "price_cop": 28000,
                        "duration_minutes": 40,
                        "description": "Lavado, corte personalizado y peinado",
                    },
                    {
                        "name": "Arreglo de Barba con Toalla Caliente",
                        "price_cop": 18000,
                        "duration_minutes": 30,
                        "description": "Perfilado con navaja y tratamiento hidratante",
                    },
                    {
                        "name": "Limpieza Facial Profunda e Hidratación",
                        "price_cop": 75000,
                        "duration_minutes": 60,
                        "description": "Extracción de impurezas, vapor de ozono y mascarilla",
                    },
                    {
                        "name": "Masaje Relajante Antiestrés",
                        "price_cop": 95000,
                        "duration_minutes": 50,
                        "description": "Masaje corporal con aceites esenciales",
                    },
                    {
                        "name": "Manicura y Pedicura Spa",
                        "price_cop": 45000,
                        "duration_minutes": 60,
                        "description": "Exfoliación, limado y esmaltado tradicional o semipermanente",
                    },
                ],
                "schedule_weekdays": "Lunes a Viernes: 9:00 AM - 7:00 PM",
                "schedule_saturdays": "Sábados: 9:00 AM - 7:00 PM",
                "holiday_policy": "Festivos: Atención de 10:00 AM a 2:00 PM con reserva previa",
                "cancellation_policy": "Cancelar o reprogramar con mínimo 2 horas de anticipación",
                "tolerance_minutes": 15,
                "payment_methods": [
                    "Nequi / Daviplata",
                    "Transferencia",
                    "Tarjetas débito/crédito",
                    "Efectivo",
                ],
                "faqs": [
                    {
                        "question": "¿Debo traer algo a mi sesión de spa?",
                        "answer": "No es necesario, te suministramos toalla y bata desechable para tu comodidad.",
                    },
                    {
                        "question": "¿Se puede pagar al llegar?",
                        "answer": "Sí, recibimos pagos por Nequi, Daviplata, tarjetas y efectivo al momento de tu atención.",
                    },
                ],
            }
        elif any(k in ind for k in ("legal", "abogad", "jurídic", "ley", "notar", "tributar")):
            return {
                "agent_name": "Martín",
                "tone": "Formal, profesional, confidencial y respetuoso",
                "services": [
                    {
                        "name": "Consulta Jurídica y Diagnóstico Inicial",
                        "price_cop": 100000,
                        "duration_minutes": 45,
                        "description": "Evaluación del caso y viabilidad jurídica",
                    },
                    {
                        "name": "Revisión y Redacción de Contratos",
                        "price_cop": 220000,
                        "duration_minutes": 60,
                        "description": "Análisis de cláusulas comerciales, laborales o de arrendamiento",
                    },
                    {
                        "name": "Asesoría Laboral y Pensional",
                        "price_cop": 150000,
                        "duration_minutes": 60,
                        "description": "Estudio de liquidaciones, fueros y derechos laborales",
                    },
                    {
                        "name": "Trámites y Representación Legal",
                        "price_cop": 350000,
                        "duration_minutes": 60,
                        "description": "Acompañamiento en diligencias o audiencias de conciliación",
                    },
                ],
                "schedule_weekdays": "Lunes a Viernes: 8:00 AM - 5:30 PM",
                "schedule_saturdays": "Sábados: 9:00 AM - 12:00 PM (Solo virtual)",
                "holiday_policy": "Festivos: Cerrado",
                "cancellation_policy": "Avisar con mínimo 4 horas de anticipación",
                "tolerance_minutes": 15,
                "payment_methods": ["Transferencia Bancolombia", "Nequi", "Tarjeta de Crédito"],
                "faqs": [
                    {
                        "question": "¿La consulta puede ser virtual?",
                        "answer": "Sí, disponemos de sesiones presenciales o por Google Meet según tu preferencia.",
                    }
                ],
            }
        elif any(k in ind for k in ("taller", "mecánic", "auto", "moto", "vehícul", "freno")):
            return {
                "agent_name": "Andrés",
                "tone": "Claro, técnico, confiable y resolutivo",
                "services": [
                    {
                        "name": "Diagnóstico y Escáner Computarizado",
                        "price_cop": 60000,
                        "duration_minutes": 45,
                        "description": "Lectura de códigos de falla y revisión electrónica",
                    },
                    {
                        "name": "Cambio de Aceite y Filtros",
                        "price_cop": 140000,
                        "duration_minutes": 45,
                        "description": "Aceite sintético/semisintético y filtro de motor",
                    },
                    {
                        "name": "Mantenimiento Preventivo de Frenos",
                        "price_cop": 95000,
                        "duration_minutes": 60,
                        "description": "Limpieza, graduación y cambio de pastillas",
                    },
                    {
                        "name": "Alineación y Balanceo",
                        "price_cop": 50000,
                        "duration_minutes": 40,
                        "description": "Alineación láser 3D y balanceo de 4 ruedas",
                    },
                ],
                "schedule_weekdays": "Lunes a Viernes: 7:30 AM - 5:30 PM",
                "schedule_saturdays": "Sábados: 8:00 AM - 2:00 PM",
                "holiday_policy": "Festivos: Cerrado",
                "cancellation_policy": "Avisar con mínimo 2 horas de anticipación",
                "tolerance_minutes": 15,
                "payment_methods": ["Nequi", "Daviplata", "Bancolombia", "Tarjetas", "Efectivo"],
                "faqs": [
                    {
                        "question": "¿Debo dejar el carro todo el día?",
                        "answer": "Para servicios rápidos de mantenimiento o escáner puedes esperar en nuestra sala de clientes (toma entre 40 y 60 min).",
                    }
                ],
            }
        elif any(k in ind for k in ("restauran", "comida", "gastronom", "café", "bar", "pizza")):
            return {
                "agent_name": "Mateo",
                "tone": "Cálido, anfitrión, alegre y servicial",
                "services": [
                    {
                        "name": "Reserva de Mesa Estándar",
                        "price_cop": 0,
                        "duration_minutes": 90,
                        "description": "Mesa reservada hasta para 6 personas",
                    },
                    {
                        "name": "Reserva Cumpleaños / Celebración",
                        "price_cop": 0,
                        "duration_minutes": 120,
                        "description": "Mesa decorada y postre de cortesía para el festejado",
                    },
                    {
                        "name": "Menú Degustación Ejecutivo",
                        "price_cop": 35000,
                        "duration_minutes": 60,
                        "description": "Entrada, plato fuerte, postre y bebida del día",
                    },
                ],
                "schedule_weekdays": "Lunes a Viernes: 12:00 PM - 10:00 PM",
                "schedule_saturdays": "Sábados: 12:00 PM - 11:00 PM",
                "holiday_policy": "Festivos: Atención normal (12:00 PM - 8:00 PM)",
                "cancellation_policy": "Avisar cancelación con al menos 1 hora de anticipación",
                "tolerance_minutes": 15,
                "payment_methods": ["Tarjetas débito/crédito", "Nequi", "Daviplata", "Efectivo"],
                "faqs": [
                    {
                        "question": "¿Tienen opciones vegetarianas?",
                        "answer": "Sí, disponemos de opciones vegetarianas y libres de gluten en nuestra carta.",
                    }
                ],
            }
        else:
            return {
                "agent_name": "Valentina",
                "tone": "Profesional, atento, resolutivo y cercano",
                "services": [
                    {
                        "name": "Sesión de Asesoría y Diagnóstico",
                        "price_cop": 50000,
                        "duration_minutes": 45,
                        "description": "Revisión inicial del requerimiento y propuesta",
                    },
                    {
                        "name": "Servicio Estándar de Atención",
                        "price_cop": 110000,
                        "duration_minutes": 60,
                        "description": "Atención especializada completa",
                    },
                    {
                        "name": "Servicio Premium Personalizado",
                        "price_cop": 190000,
                        "duration_minutes": 90,
                        "description": "Solución integral con seguimiento prioritario",
                    },
                ],
                "schedule_weekdays": "Lunes a Viernes: 8:00 AM - 6:00 PM",
                "schedule_saturdays": "Sábados: 9:00 AM - 1:00 PM",
                "holiday_policy": "Festivos solo atención hasta mediodía (8:00 AM - 1:00 PM)",
                "cancellation_policy": "Avisar con mínimo 2 horas de anticipación",
                "tolerance_minutes": 15,
                "payment_methods": [
                    "Nequi / Daviplata",
                    "Transferencia Bancolombia",
                    "Tarjetas",
                    "Efectivo",
                ],
                "faqs": [
                    {
                        "question": "¿Cómo confirmo mi cita?",
                        "answer": "Tu cita queda registrada en nuestro calendario inmediatamente acordemos la hora y tus datos de contacto.",
                    }
                ],
            }

    async def ai_enrich_business(
        self,
        req: AIEnrichRequest,
    ) -> AIEnrichResponse:
        """
        Uses Gemini 3.8 Flash to intelligently enrich business context with realistic
        services, market pricing in COP, operating schedules, holiday policies,
        cancellation rules, and a production-grade Master System Prompt.
        """
        system_instruction = (
            "Eres el principal consultor de operaciones, agendamiento y experiencia de cliente "
            f"para WhatsApp en {req.city_or_country or 'Colombia'}. "
            "Tu misión es estructurar el perfil operativo de un negocio para configurar su agente de inteligencia artificial. "
            "Debes proponer servicios con precios reales de mercado en pesos colombianos (COP), "
            "horarios lógicos adaptados a la cultura local (incluyendo política de días festivos), "
            "tolerancia de llegada, formas de pago y preguntas frecuentes típicas del sector. "
            "Responde ÚNICAMENTE con un JSON válido sin texto adicional ni bloques decorativos."
        )

        user_content = (
            f"Negocio: {req.business_name}\n"
            f"Sector/Industria: {req.industry}\n"
            f"Ubicación: {req.city_or_country or 'Colombia'}\n"
            f"{f'Detalles adicionales: {req.extra_notes}' if req.extra_notes else ''}\n\n"
            "Genera el JSON con esta estructura exacta:\n"
            "{\n"
            '  "agent_name": "Nombre empático y profesional para el asistente (ej: Sofía, Carlos, Valentina)",\n'
            '  "tone": "Cálido, profesional, empático y resolutivo",\n'
            '  "services": [\n'
            '    {"name": "Servicio 1", "price_cop": 80000, "duration_minutes": 45, "description": "Breve descripción"},\n'
            '    {"name": "Servicio 2", "price_cop": 120000, "duration_minutes": 60, "description": "Breve descripción"},\n'
            '    {"name": "Servicio 3", "price_cop": 50000, "duration_minutes": 30, "description": "Breve descripción"}\n'
            '  ],\n'
            '  "schedule_weekdays": "Lunes a Viernes: 8:00 AM - 6:00 PM",\n'
            '  "schedule_saturdays": "Sábados: 9:00 AM - 1:00 PM",\n'
            '  "holiday_policy": "Festivos solo atención hasta mediodía (8:00 AM - 1:00 PM)",\n'
            '  "cancellation_policy": "Cancelar o reprogramar con mínimo 2 horas de anticipación",\n'
            '  "tolerance_minutes": 15,\n'
            '  "payment_methods": ["Nequi / Daviplata", "Transferencia Bancolombia", "Tarjetas débito/crédito", "Efectivo"],\n'
            '  "faqs": [\n'
            '    {"question": "¿Cuáles son los métodos de pago?", "answer": "Aceptamos transferencias por Nequi, Daviplata, Bancolombia, tarjetas de crédito/débito y efectivo."},\n'
            '    {"question": "¿Se requiere preparación previa para la cita?", "answer": "Recomendamos llegar 10 minutos antes de la hora acordada."}\n'
            '  ]\n'
            "}"
        )

        fallback_data = self._get_fallback_enrichment(req)
        enriched_data = None

        try:
            provider = LLMProviderFactory().get_provider(
                "google", "gemini-3.8-flash", system_instruction
            )
            resp = await asyncio.wait_for(
                provider.generate(
                    system_prompt=system_instruction,
                    messages=[LLMConversationMessage(role="user", content=user_content)],
                    temperature=0.2,
                    max_tokens=2048,
                ),
                timeout=8.0,
            )
            raw = (resp.content or "").strip()
            clean_json = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
            clean_json = re.sub(r"^```\s*$", "", clean_json, flags=re.MULTILINE).strip()
            m = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict) and "services" in parsed:
                    enriched_data = parsed
        except Exception as exc:
            logger.warning("AI enrichment via LLM failed, using curated niche profile: %s", exc)

        data = enriched_data or fallback_data

        service_items: list[ServiceItem] = []
        for s in data.get("services", []):
            if isinstance(s, dict) and s.get("name"):
                service_items.append(
                    ServiceItem(
                        name=str(s["name"]),
                        price_cop=int(s["price_cop"]) if s.get("price_cop") is not None else None,
                        duration_minutes=int(s.get("duration_minutes") or 45),
                        description=s.get("description"),
                    )
                )
        if not service_items:
            service_items = [
                ServiceItem(
                    name="Consulta inicial y diagnóstico", price_cop=60000, duration_minutes=45
                ),
                ServiceItem(
                    name="Servicio estándar de atención", price_cop=120000, duration_minutes=60
                ),
            ]

        full_schedule = f"{data.get('schedule_weekdays', 'Lunes a Viernes: 8:00 AM - 6:00 PM')}\n{data.get('schedule_saturdays', 'Sábados: 9:00 AM - 1:00 PM')}"

        temp_answers = OnboardingAnswers(
            business_name=req.business_name,
            business_type=req.industry,
            city_or_country=req.city_or_country,
            agent_name=data.get("agent_name", "Sofía"),
            tone=data.get("tone", "profesional, cálido y servicial"),
            services=service_items,
            schedule=full_schedule,
            holiday_policy=data.get(
                "holiday_policy", "Días festivos: atención hasta mediodía (8:00 AM - 1:00 PM)"
            ),
            cancellation_policy_hours=2,
            cancellationPolicy=data.get(
                "cancellation_policy", "Avisar con mínimo 2 horas de anticipación"
            ),
            tolerance_minutes=int(data.get("tolerance_minutes") or 15),
            payment_methods=data.get(
                "payment_methods", ["Nequi / Daviplata", "Tarjetas", "Efectivo"]
            ),
            faqs=data.get("faqs", []),
            use_emojis=True,
        )
        master_prompt = _build_system_prompt(temp_answers)

        return AIEnrichResponse(
            agent_name=data.get("agent_name", "Sofía"),
            tone=data.get("tone", "profesional, cálido y servicial"),
            services=service_items,
            schedule_weekdays=data.get("schedule_weekdays", "Lunes a Viernes: 8:00 AM - 6:00 PM"),
            schedule_saturdays=data.get("schedule_saturdays", "Sábados: 9:00 AM - 1:00 PM"),
            holiday_policy=data.get(
                "holiday_policy", "Festivos solo atención hasta mediodía (8:00 AM - 1:00 PM)"
            ),
            cancellation_policy=data.get(
                "cancellation_policy", "Cancelar o reprogramar con mínimo 2 horas de anticipación"
            ),
            tolerance_minutes=int(data.get("tolerance_minutes") or 15),
            payment_methods=data.get(
                "payment_methods",
                ["Nequi / Daviplata", "Transferencia Bancolombia", "Tarjetas", "Efectivo"],
            ),
            faqs=data.get("faqs", []),
            recommended_system_prompt=master_prompt,
        )


# ---------------------------------------------------------------------------
# Prompt engineering engine (Section 6 of project documentation)
# ---------------------------------------------------------------------------


def _is_onboarding_complete(answers: dict) -> bool:
    """
    Validate that minimum required onboarding fields are present.

    Required fields for a valid system prompt:
    - business_name or companyName
    - business_type or industry
    - services (at least one)
    - hours or schedule (at least one)
    - agent_name or agentName
    """
    return bool(
        (answers.get("business_name") or answers.get("companyName"))
        and (answers.get("business_type") or answers.get("industry"))
        and answers.get("services")
        and (answers.get("hours") or answers.get("schedule"))
        and (answers.get("agent_name") or answers.get("agentName"))
    )


def _build_services_block(answers: OnboardingAnswers) -> str:
    """Format the services list for the system prompt."""
    if not answers.services:
        return "- (No services configured)"

    if isinstance(answers.services, str):
        return answers.services.strip()

    lines: list[str] = []
    for svc in answers.services:
        price_part = ""
        if svc.price_cop is not None:
            price_part = f" (${svc.price_cop:,} COP)"
        duration_part = ""
        if svc.duration_minutes is not None:
            duration_part = f" — {svc.duration_minutes} min"
        lines.append(f"- {svc.name}{price_part}{duration_part}")
    return "\n".join(lines)


def _build_hours_block(answers: OnboardingAnswers) -> str:
    """Format the operating hours for the system prompt."""
    if isinstance(answers.hours, str) and answers.hours.strip():
        return answers.hours.strip()
    if answers.schedule and answers.schedule.strip() and not answers.hours:
        return answers.schedule.strip()
    if not answers.hours:
        return "- (No hours configured)"

    day_names = {
        "monday": "Lunes",
        "tuesday": "Martes",
        "wednesday": "Miércoles",
        "thursday": "Jueves",
        "friday": "Viernes",
        "saturday": "Sábado",
        "sunday": "Domingo",
    }
    lines: list[str] = []
    for day_key, hours in answers.hours.items():
        day_label = day_names.get(day_key, day_key.capitalize())
        lines.append(f"- {day_label}: {hours.open} - {hours.close}")
    return "\n".join(lines)


def _build_system_prompt(answers: OnboardingAnswers) -> str:
    """
    Assemble the Master System Prompt from onboarding answers.

    Follows the 6-section structure defined in Section 6 of the project doc,
    enriched with holiday operations, payment methods, arrival tolerance,
    and the strict Google Calendar booking protocol.
    """
    name = answers.business_name or answers.companyName or "el negocio"
    agent_name = answers.agent_name or answers.agentName or "Asistente"
    business_type = answers.business_type or answers.industry or "negocio"
    tone = answers.tone or "profesional, cálido y resolutivo"
    emoji_rule = (
        "Usa emojis con moderación (máximo 1 o 2 por mensaje, acorde a WhatsApp)."
        if answers.use_emojis
        else "No uses emojis."
    )
    cancel_hours = (
        f"Se deben hacer con mínimo {answers.cancellation_policy_hours} horas de anticipación."
        if answers.cancellation_policy_hours is not None
        else (
            answers.cancellationPolicy
            or "Avisar cancelación o reprogramación con mínimo 2 horas de anticipación."
        )
    )
    holiday_rule = (
        answers.holiday_policy
        or "Días festivos: atención reducida hasta mediodía (8:00 AM - 1:00 PM) o cerrado."
    )
    tolerance_rule = f"{answers.tolerance_minutes or 15} minutos de tolerancia de espera antes de liberar el turno."

    # Payment methods
    if isinstance(answers.payment_methods, list) and answers.payment_methods:
        payment_text = ", ".join(answers.payment_methods)
    elif isinstance(answers.payment_methods, str) and answers.payment_methods.strip():
        payment_text = answers.payment_methods.strip()
    else:
        payment_text = (
            "Nequi / Daviplata, Transferencia Bancolombia, Tarjetas débito/crédito y Efectivo."
        )

    # FAQs
    faqs_block = ""
    if answers.faqs and isinstance(answers.faqs, list):
        faq_lines = [
            f"- P: {f.get('question', '')}\n  R: {f.get('answer', '')}"
            for f in answers.faqs
            if isinstance(f, dict) and f.get("question")
        ]
        if faq_lines:
            faqs_block = "\n\n**Preguntas Frecuentes:**\n" + "\n".join(faq_lines)

    emergency_line = (
        "SÍ atendemos urgencias. Comunicarse de inmediato a nuestra línea de atención prioritaria."
        if answers.handles_emergencies
        else "NO atendemos urgencias médicas/críticas sin cita previa."
    )
    escalation_line = (
        f"Escala al equipo de {name} si el cliente tiene una consulta fuera de catálogo, "
        f"si hay una queja, o si solicita hablar con un asesor humano. "
        f"Contacto de escalación: {answers.escalation_contact or answers.whatsappPhone or answers.phone or 'nuestra línea principal'}."
    )

    prompt = f"""## 1. ROL Y ALCANCE
Eres {agent_name}, el asistente virtual de {name}. Tu objetivo es atender a los clientes por WhatsApp con empatía y calidez, brindar información sobre nuestros servicios y agendar sus citas en Google Calendar. Eres un asistente especializado en {business_type}.

## 2. DATOS DEL NEGOCIO
**Nombre:** {name}
**Sector:** {business_type}
{f"**Ubicación:** {answers.address or answers.city_or_country}" if (answers.address or answers.city_or_country) else ""}

**Catálogo de Servicios y Tarifas (en pesos colombianos COP):**
{_build_services_block(answers)}

**Horario de Atención:**
{_build_hours_block(answers)}
- **Días Festivos:** {holiday_rule}

**Políticas de Operación y Pagos:**
- **Cancelaciones y Cambios:** {cancel_hours}
- **Tolerancia de Llegada:** {tolerance_rule}
- **Medios de Pago Aceptados:** {payment_text}
- **Urgencias:** {emergency_line}{faqs_block}

## 3. REGLAS DE ATENCIÓN Y GUARDRAILS
1. NUNCA inventes precios, horarios o servicios que no estén listados en el catálogo. Si no sabes un dato, dilo con amabilidad y ofrece consultar con el equipo humano.
2. Comunícate en un tono {tone}. Máximo 2 a 3 oraciones breves por mensaje, ideal para lectura ágil en WhatsApp.
3. {emoji_rule}
4. Ante preguntas completamente ajenas al negocio, responde con cortesía: "Solo puedo ayudarte con información y citas para {name}. ¿Te gustaría que revisemos disponibilidad?"

## 4. PROTOCOLO OBLIGATORIO DE AGENDAMIENTO (GOOGLE CALENDAR)
- Si el cliente pregunta horarios, ofrece opciones concretas dentro del horario laboral (ej: 10:00 AM o 3:30 PM).
- Si el cliente pregunta por un festivo, recuerda amablemente: "{holiday_rule}".
- En cuanto el cliente elija o confirme la fecha y hora de su cita, confirma con entusiasmo la reserva y coloca SIEMPRE al final exacto de tu mensaje el comando:
[[AGENDAR|YYYY-MM-DD|HH:MM|Nombre del cliente|Servicio]]

## 5. ESCALACIÓN HUMANA
{escalation_line}
""".strip()

    return prompt
