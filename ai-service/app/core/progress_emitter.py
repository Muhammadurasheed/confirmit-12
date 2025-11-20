"""
Real-time Progress Emitter for AI Agents
Emits detailed progress updates to Firebase for backend consumption
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
from app.core.firebase import db

logger = logging.getLogger(__name__)


def _sanitize_for_firebase(obj: Any) -> Any:
    """
    Recursively sanitize data for Firebase Firestore serialization.
    Converts numpy types, removes nested objects, handles arrays/lists.
    """
    if obj is None:
        return None
    
    # Handle numpy types
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        return val if not (np.isnan(val) or np.isinf(val)) else 0.0
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        # Convert arrays to lists (limit size for Firebase)
        if obj.size > 1000:
            return f"[Array of {obj.shape}]"
        return [_sanitize_for_firebase(item) for item in obj.tolist()]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: _sanitize_for_firebase(v) for k, v in obj.items()}
    
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_firebase(item) for item in obj]
    
    # Primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Unknown types - convert to string representation
    return str(obj)


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
                # Sanitize details to ensure Firebase compatibility
                progress_update['details'] = _sanitize_for_firebase(details)
            
            # Update Firebase with latest progress
            self.progress_ref.update({
                'current_progress': progress_update,
                'last_updated': datetime.utcnow().isoformat(),
            })
            
            logger.info(f"📡 [{self.receipt_id}] {agent}: {message} ({progress}%)")
            
        except Exception as e:
            logger.error(f"❌ Failed to emit progress: {str(e)}")
            # Don't fail the analysis if progress emission fails
