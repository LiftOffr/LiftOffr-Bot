import json
import asyncio
import threading
import logging
import requests

try:
    import websockets
except ImportError:
    # If websockets is not available, create a mock class
    class MockWebsockets:
        def __init__(self):
            class MockConnect:
                def __call__(self, *args, **kwargs):
                    raise ImportError("websockets module not installed. Please install with 'pip install websockets'")
                
                @staticmethod
                async def __aenter__(*args, **kwargs):
                    raise ImportError("websockets module not installed. Please install with 'pip install websockets'")
                
                @staticmethod
                async def __aexit__(*args, **kwargs):
                    pass
            
            self.connect = MockConnect()
    
    websockets = MockWebsockets()

logger = logging.getLogger(__name__)

class SimpleWebSocketClient:
    """A simpler WebSocket client implementation using websockets library"""
    
    def __init__(self, url, on_message=None, on_error=None, on_close=None, on_open=None):
        self.url = url
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.on_open = on_open
        self.ws = None
        self.running = False
        self.thread = None
        self.loop = None
        self.message_queue = asyncio.Queue()
    
    def send(self, data):
        """Send data to the WebSocket"""
        if self.running and self.loop:
            asyncio.run_coroutine_threadsafe(self._send(data), self.loop)
    
    async def _send(self, data):
        """Async method to send data to the WebSocket"""
        if self.ws:
            await self.ws.send(data)
    
    def close(self):
        """Close the WebSocket connection"""
        self.running = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._close(), self.loop)
    
    async def _close(self):
        """Async method to close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
    
    async def _connect(self):
        """Async method to connect to the WebSocket"""
        try:
            async with websockets.connect(self.url) as websocket:
                self.ws = websocket
                logger.info(f"Connected to WebSocket: {self.url}")
                
                if self.on_open:
                    self.on_open(self)
                
                self.running = True
                
                # Start a task to process outgoing messages
                asyncio.create_task(self._process_outgoing_messages())
                
                # Process incoming messages
                async for message in websocket:
                    if self.on_message:
                        self.on_message(self, message)
            
            logger.info("WebSocket connection closed")
            
            if self.on_close:
                self.on_close(self, None, None)
        
        except Exception as e:
            if self.on_error:
                self.on_error(self, e)
            logger.error(f"WebSocket error: {e}")
        
        finally:
            self.running = False
    
    async def _process_outgoing_messages(self):
        """Process outgoing messages from the queue"""
        while self.running:
            try:
                message = await self.message_queue.get()
                await self.ws.send(message)
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                await asyncio.sleep(0.1)
    
    def _run_forever(self):
        """Run the WebSocket client in a separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect())
        finally:
            self.loop.close()
            self.loop = None
    
    def run_forever(self):
        """Start the WebSocket client in a separate thread"""
        self.thread = threading.Thread(target=self._run_forever)
        self.thread.daemon = True
        self.thread.start()