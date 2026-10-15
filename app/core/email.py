"""
Email delivery service for Zolution.
Supports Resend API (HTTP), SMTP (Gmail, Hostinger, SendGrid), and development console simulation.
"""

import email.message
import logging
import smtplib
from typing import Optional

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def build_verification_html(code: str, company_name: Optional[str] = None) -> str:
    """Generate modern, responsive HTML for the verification email."""
    name_display = f"para <strong>{company_name}</strong>" if company_name else ""
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f4f4f5; margin: 0; padding: 20px; }}
          .container {{ max-width: 520px; margin: 0 auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }}
          .header {{ background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 32px 24px; text-align: center; color: #ffffff; }}
          .logo {{ display: inline-block; width: 44px; height: 44px; line-height: 44px; background: rgba(255,255,255,0.2); border-radius: 12px; font-weight: 800; font-size: 24px; margin-bottom: 8px; }}
          .content {{ padding: 32px 28px; color: #18181b; }}
          .code-box {{ margin: 24px 0; background: #f4f4f9; border: 2px dashed #6366f1; border-radius: 12px; text-align: center; padding: 18px; }}
          .code {{ font-family: monospace; font-size: 34px; font-weight: 800; letter-spacing: 6px; color: #4338ca; }}
          .footer {{ padding: 20px; text-align: center; font-size: 12px; color: #71717a; border-top: 1px solid #f4f4f5; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <div class="logo">Z</div>
            <h1 style="margin: 0; font-size: 22px;">Zolution AI Platform</h1>
            <p style="margin: 6px 0 0; opacity: 0.9; font-size: 13px;">Verificación de cuenta y seguridad</p>
          </div>
          <div class="content">
            <h2 style="font-size: 18px; margin-top: 0;">Tu código de verificación</h2>
            <p style="color: #52525b; font-size: 14px; line-height: 1.5;">
              Has solicitado registrarte o iniciar sesión {name_display} en Zolution. Usa el siguiente código de 6 dígitos para verificar tu cuenta:
            </p>
            <div class="code-box">
              <div class="code">{code}</div>
            </div>
            <p style="color: #71717a; font-size: 12px; line-height: 1.4;">
              Este código es de un solo uso y vencerá en 10 minutos. Si tú no realizaste esta solicitud, puedes ignorar este mensaje de forma segura.
            </p>
          </div>
          <div class="footer">
            © Zolution AI · Automatización Inteligente con WhatsApp y Google Calendar
          </div>
        </div>
      </body>
    </html>
    """


async def send_verification_email(
    to_email: str,
    code: str,
    company_name: Optional[str] = None,
) -> bool:
    """
    Dispatch a verification email with the 6-digit code.
    Tries Resend first, then SMTP, and falls back to server console logging.
    """
    settings = get_settings()
    subject = f"{code} es tu código de verificación de Zolution"
    html_content = build_verification_html(code, company_name)
    plain_content = f"Tu código de verificación de Zolution es: {code}. Vence en 10 minutos."

    # 1. Try Resend API (HTTP)
    if settings.RESEND_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    "https://api.resend.com/emails",
                    headers={
                        "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "from": settings.EMAIL_FROM,
                        "to": [to_email],
                        "subject": subject,
                        "html": html_content,
                        "text": plain_content,
                    },
                )
                if res.status_code in (200, 201):
                    logger.info("Verification email sent via Resend to %s", to_email)
                    return True
                logger.error("Resend API error (%d): %s", res.status_code, res.text)
        except Exception as exc:
            logger.error("Failed sending via Resend: %s", exc)

    # 2. Try SMTP if configured
    if settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD:
        try:
            msg = email.message.EmailMessage()
            msg["Subject"] = subject
            msg["From"] = settings.SMTP_USER
            msg["To"] = to_email
            msg.set_content(plain_content)
            msg.add_alternative(html_content, subtype="html")

            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.send_message(msg)

            logger.info("Verification email sent via SMTP to %s", to_email)
            return True
        except Exception as exc:
            logger.error("Failed sending via SMTP: %s", exc)

    # 3. Local development only: no provider configured and not in production.
    #    The code is printed in the SERVER log — never returned to the browser.
    provider_configured = bool(settings.RESEND_API_KEY) or bool(
        settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD
    )
    if not provider_configured and settings.APP_ENV != "production":
        logger.warning(
            "[ZOLUTION DEV EMAIL] To: %s | OTP: %s (configure RESEND_API_KEY for real delivery)",
            to_email,
            code,
        )
        return True

    # A provider was configured but failed, or production has no provider: report failure.
    logger.error("Verification email could not be delivered to %s", to_email)
    return False
