"""
Real-time Progress Emitter for AI Agents
Emits detailed progress updates to Firebase for backend consumption
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from app.core.firebase import db

logger = logging.getLogger(__name__)


class ProgressEmitter:
    """Emits detailed agent progress to Firebase Firestore"""
    
    def __init__(self, receipt_id: str):
        self.receipt_id = receipt_id
        self.progress_ref = db.collection('receipts').document(receipt_id)
        
    async def emit(
        self,
        agent: str,
        stage: str,
        message: str,
        progress: int,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Emit progress update to Firebase
        
        Args:
            agent: Name of agent (vision, forensic, metadata, reputation, reasoning)
            stage: Current stage (ocr_started, forensics_running, etc.)
            message: User-friendly message describing what's happening
            progress: Progress percentage (0-100)
            details: Optional dict with specific details (e.g., extracted data)
        """
        try:
            progress_update = {
                'agent': agent,
                'stage': stage,
                'message': message,
                'progress': progress,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            if details:
                progress_update['details'] = details
            
            # Update Firebase with latest progress
            self.progress_ref.update({
                'current_progress': progress_update,
                'last_updated': datetime.utcnow().isoformat(),
            })
            
            logger.info(f"📡 [{self.receipt_id}] {agent}: {message} ({progress}%)")
            
        except Exception as e:
            logger.error(f"❌ Failed to emit progress: {str(e)}")
            # Don't fail the analysis if progress emission fails
