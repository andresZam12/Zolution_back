"""
Database seed script for Zolution multi-tenant AI platform.

Populates representative organizations across multiple business industries
(Legal Services, IT Consulting, Healthcare/Clinics, Wellness & Spas) with
configured AI agents, onboarding answers, and sample WhatsApp conversations.

Usage:
    cd backend
    python -m scripts.seed_data
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal, engine, Base
from app.tenants.models import Organization, User
from app.agents.models import AgentConfig
from app.conversations.models import Conversation, Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEMO_TENANTS = [
    {
        "name": "Lex & Co. Asesoría Legal",
        "slug": "lex-asesores",
        "plan": "professional",
        "status": "active",
        "user_email": "admin@lexasesores.com",
        "auth0_id": "auth0|demo_lex_001",
        "whatsapp_phone": "+573001112233",
        "industry": "Servicios Legales y Jurídicos",
        "agent": {
            "name": "Martín",
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "tone": "Formal, preciso y confidencial",
            "services": (
                "1. Consulta inicial y revisión contractual (1 hora)\n"
                "2. Asesoría corporativa y societaria (45 mins)\n"
                "3. Diagnóstico de litigios civiles y comerciales (1 hora)"
            ),
            "schedule": "Lunes a Viernes: 8:30 AM - 5:30 PM",
            "policy": "Reagendar o cancelar con al menos 4 horas de anticipación.",
            "prompt": (
                "Eres Martín, el asistente legal virtual de Lex & Co. Asesoría Legal. "
                "Tu objetivo es brindar información preliminar sobre nuestros servicios jurídicos, "
                "filtrar el tipo de caso del cliente y coordinar citas con los abogados especialistas "
                "utilizando Google Calendar mediante las herramientas disponibles ('check_availability' y 'book_appointment'). "
                "Mantén un tono siempre profesional, discreto y empático."
            ),
        },
        "sample_chat": {
            "customer_name": "Dr. Fernando Gómez",
            "customer_phone": "+573105556677",
            "messages": [
                {"role": "user", "content": "Hola, buenas tardes. Requiero agendar una asesoría para revisar un contrato de fusión empresarial."},
                {"role": "assistant", "content": "Buenas tardes, Dr. Gómez. Con gusto le asisto. Contamos con consultas especializadas en derecho corporativo y revisión de contratos de 1 hora. ¿Para qué fecha u horario le convendría la sesión?"},
                {"role": "user", "content": "¿Tienen disponibilidad este viernes a las 10:00 AM?"},
                {"role": "assistant", "content": "Permítame validar la agenda de nuestro equipo corporativo para este viernes a las 10:00 AM... Perfecto, tenemos ese espacio disponible. ¿Desea que reserve la cita a su nombre?"},
            ],
        },
    },
    {
        "name": "NexTech Soluciones Cloud",
        "slug": "nextech-cloud",
        "plan": "enterprise",
        "status": "active",
        "user_email": "cto@nextechcloud.io",
        "auth0_id": "auth0|demo_nextech_002",
        "whatsapp_phone": "+573002223344",
        "industry": "Tecnología, Cloud & Soporte TI",
        "agent": {
            "name": "Alex",
            "provider": "google",
            "model": "gemini-2.5-flash",
            "tone": "Técnico, dinámico, consultivo y ágil",
            "services": (
                "1. Diagnóstico de arquitectura AWS/GCP (45 mins)\n"
                "2. Auditoría DevSecOps y seguridad cloud (1 hora)\n"
                "3. Sesión discovery de desarrollo a medida (30 mins)"
            ),
            "schedule": "Lunes a Viernes: 8:00 AM - 6:00 PM",
            "policy": "Confirmación inmediata con enlace a Google Meet generado en Calendar.",
            "prompt": (
                "Eres Alex, tech lead y asistente comercial de NexTech Soluciones Cloud. "
                "Ayudas a directores de tecnología y fundadores a entender nuestras soluciones de infraestructura cloud "
                "y coordinas llamadas técnicas discovery mediante 'check_availability' y 'book_appointment'. "
                "Sé conciso, técnico y orientado a resultados."
            ),
        },
        "sample_chat": {
            "customer_name": "Laura Méndez (Fintech Co)",
            "customer_phone": "+573128889900",
            "messages": [
                {"role": "user", "content": "Hola NexTech, estamos evaluando migrar nuestra infraestructura a Kubernetes y queremos asesoría."},
                {"role": "assistant", "content": "¡Hola Laura! Excelente iniciativa. Nuestras sesiones de arquitectura Cloud y DevSecOps ayudan a dimensionar clústeres y optimizar costos. Podemos coordinar un discovery técnico de 45 mins. ¿Qué día les funciona mejor?"},
                {"role": "user", "content": "Mañana en la tarde después de las 3:00 PM."},
                {"role": "assistant", "content": "Consultando disponibilidad... Tenemos las 3:30 PM o las 4:30 PM de mañana disponibles con nuestro arquitecto cloud. ¿Cuál prefieres?"},
            ],
        },
    },
    {
        "name": "Clínica Dental & Estética Santa María",
        "slug": "dra-garcia-dental",
        "plan": "starter",
        "status": "active",
        "user_email": "citas@clinicasantamaria.com",
        "auth0_id": "auth0|demo_dental_003",
        "whatsapp_phone": "+573003334455",
        "industry": "Salud, Odontología & Estética",
        "agent": {
            "name": "Sofía",
            "provider": "openai",
            "model": "gpt-4o",
            "tone": "Cálida, tranquilizadora y resolutiva",
            "services": (
                "1. Valoración odontológica inicial y diagnóstico general (30 mins)\n"
                "2. Limpieza ultrasónica y profilaxis profunda (45 mins)\n"
                "3. Evaluación para ortodoncia invisible / diseño de sonrisa (45 mins)"
            ),
            "schedule": "Lunes a Sábado: 8:00 AM - 6:00 PM",
            "policy": "Llegar 10 minutos antes para diligenciamiento de ficha clínica.",
            "prompt": (
                "Eres Sofía, asistente virtual de la Clínica Dental & Estética Santa María. "
                "Respondes dudas sobre tratamientos dentales, horarios y precios aproximados, "
                "y agendas citas para los pacientes en Google Calendar con 'check_availability' y 'book_appointment'."
            ),
        },
        "sample_chat": {
            "customer_name": "Carlos Rodríguez",
            "customer_phone": "+573151112233",
            "messages": [
                {"role": "user", "content": "Buenas tardes, quisiera consultar el costo de una limpieza y agendar para el sábado."},
                {"role": "assistant", "content": "¡Hola Carlos! Un gusto saludarte. La limpieza ultrasónica profunda tiene un valor de $120.000 COP e incluye remoción de sarro y pulido dental. ¿Prefieres el sábado en la mañana o en la tarde?"},
                {"role": "user", "content": "En la mañana tipo 10:00 AM estaría genial."},
                {"role": "assistant", "content": "Revisando agenda del sábado... El cupo de las 10:00 AM está libre con la Dra. García. ¿Confirmo tu cita con tu número de teléfono?"},
            ],
        },
    },
    {
        "name": "Luxe Wellness & Spa",
        "slug": "luxe-spa",
        "plan": "starter",
        "status": "active",
        "user_email": "reservas@luxespa.co",
        "auth0_id": "auth0|demo_spa_004",
        "whatsapp_phone": "+573004445566",
        "industry": "Bienestar, Spa & Belleza",
        "agent": {
            "name": "Camila",
            "provider": "anthropic",
            "model": "claude-haiku-4-5",
            "tone": "Relajante, elegante y atenta",
            "services": (
                "1. Masaje relajante descontracturante con aromaterapia (60 mins)\n"
                "2. Circuito de hidroterapia y sauna termal (90 mins)\n"
                "3. Ritual facial rejuvenecedor con colágeno (60 mins)"
            ),
            "schedule": "Martes a Domingo: 9:00 AM - 7:00 PM",
            "policy": "Depósito del 20% para confirmar reserva de fines de semana.",
            "prompt": (
                "Eres Camila, la concierge de bienestar de Luxe Wellness & Spa. "
                "Ayudas a los clientes a elegir el paquete ideal de relajación y agendas sus "
                "citas de spa en Google Calendar utilizando 'check_availability' y 'book_appointment'."
            ),
        },
        "sample_chat": {
            "customer_name": "Mariana Vélez",
            "customer_phone": "+573187778899",
            "messages": [
                {"role": "user", "content": "Hola! Quiero regalarle un día de spa a mi mamá por su cumpleaños."},
                {"role": "assistant", "content": "¡Qué hermoso detalle, Mariana! Para ocasiones especiales recomendamos nuestro 'Ritual Renovador' que combina masaje relajante de 60 mins con hidroterapia. ¿Para qué fecha te gustaría reservarlo?"},
                {"role": "user", "content": "Para este domingo a las 11:00 AM."},
                {"role": "assistant", "content": "Verificando cabinas para este domingo a las 11:00 AM... ¡Tenemos una cabina suite libre! ¿A qué nombre completo programamos la reserva?"},
            ],
        },
    },
]


async def seed():
    """Execute multi-industry database seeding."""
    logger.info("Starting Zolution multi-industry database seeding...")

    async with AsyncSessionLocal() as session:
        for tenant_data in DEMO_TENANTS:
            # 1. Check if organization already exists
            stmt = select(Organization).where(Organization.slug == tenant_data["slug"])
            result = await session.execute(stmt)
            org = result.scalar_one_or_none()

            if not org:
                logger.info(f"Creating organization: {tenant_data['name']} ({tenant_data['slug']})")
                org = Organization(
                    id=uuid.uuid4(),
                    name=tenant_data["name"],
                    slug=tenant_data["slug"],
                    plan=tenant_data["plan"],
                    status=tenant_data["status"],
                )
                session.add(org)
                await session.flush()
            else:
                logger.info(f"Organization already exists: {tenant_data['slug']}")

            # 2. Check/create owner user
            stmt_user = select(User).where(User.email == tenant_data["user_email"])
            user_res = await session.execute(stmt_user)
            user = user_res.scalar_one_or_none()

            if not user:
                logger.info(f"Creating owner user: {tenant_data['user_email']}")
                user = User(
                    id=uuid.uuid4(),
                    auth0_id=tenant_data["auth0_id"],
                    email=tenant_data["user_email"],
                    role="owner",
                    is_active=True,
                    organization_id=org.id,
                )
                session.add(user)
                await session.flush()

            # 3. Check/create AgentConfig
            stmt_agent = select(AgentConfig).where(AgentConfig.organization_id == org.id)
            agent_res = await session.execute(stmt_agent)
            agent_cfg = agent_res.scalar_one_or_none()

            agent_meta = tenant_data["agent"]
            onboarding_answers = {
                "company_name": tenant_data["name"],
                "industry": tenant_data["industry"],
                "whatsapp_phone": tenant_data["whatsapp_phone"],
                "agent_name": agent_meta["name"],
                "tone": agent_meta["tone"],
                "services": agent_meta["services"],
                "schedule": agent_meta["schedule"],
                "cancellation_policy": agent_meta["policy"],
            }

            if not agent_cfg:
                logger.info(f"Creating AgentConfig for org: {org.name}")
                agent_cfg = AgentConfig(
                    id=uuid.uuid4(),
                    organization_id=org.id,
                    system_prompt_generated=agent_meta["prompt"],
                    onboarding_answers=onboarding_answers,
                    llm_provider=agent_meta["provider"],
                    llm_model=agent_meta["model"],
                    status="active",
                    whatsapp_phone_number_id=tenant_data["whatsapp_phone"],
                )
                session.add(agent_cfg)
                await session.flush()
            else:
                agent_cfg.system_prompt_generated = agent_meta["prompt"]
                agent_cfg.onboarding_answers = onboarding_answers
                agent_cfg.status = "active"

            # 4. Check/create sample conversation
            chat_data = tenant_data["sample_chat"]
            stmt_conv = select(Conversation).where(
                Conversation.organization_id == org.id,
                Conversation.customer_phone == chat_data["customer_phone"],
            )
            conv_res = await session.execute(stmt_conv)
            conv = conv_res.scalar_one_or_none()

            if not conv:
                logger.info(f"Creating sample conversation for customer: {chat_data['customer_name']}")
                conv = Conversation(
                    id=uuid.uuid4(),
                    organization_id=org.id,
                    customer_phone=chat_data["customer_phone"],
                    customer_name=chat_data["customer_name"],
                    status="active",
                )
                session.add(conv)
                await session.flush()

                for msg_item in chat_data["messages"]:
                    msg = Message(
                        id=uuid.uuid4(),
                        conversation_id=conv.id,
                        organization_id=org.id,
                        role=msg_item["role"],
                        content=msg_item["content"],
                        tokens_in=50 if msg_item["role"] == "user" else 200,
                        tokens_out=150 if msg_item["role"] == "assistant" else 0,
                    )
                    session.add(msg)

        await session.commit()
        logger.info("Successfully seeded multi-industry tenants, agent configs, and demo chats.")


if __name__ == "__main__":
    asyncio.run(seed())
