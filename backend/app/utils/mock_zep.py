from zep_cloud.client import Zep as RealZep
from ..config import Config
from .local_graph_store import LocalZepClient
from .logger import get_logger

logger = get_logger('mirofish.mock_zep')

class ZepWrapper:
    """Wrapper that resolves to LocalZepClient or RealZep based on ZEP_API_KEY value."""
    def __new__(cls, api_key=None):
        key = api_key or Config.ZEP_API_KEY
        if not key or key == "local_mock" or "your_" in key or key == "mock":
            logger.info("Using Local Offline Graph Engine (LocalZepClient)")
            return LocalZepClient()
        try:
            logger.info("Initializing Real Zep Cloud Client...")
            return RealZep(api_key=key)
        except Exception as e:
            logger.warning(f"Failed to initialize Real Zep Client: {e}. Falling back to LocalZepClient.")
            return LocalZepClient()

# Alias the wrapper as Zep so the client code can do `from ..utils.mock_zep import Zep`
Zep = ZepWrapper
