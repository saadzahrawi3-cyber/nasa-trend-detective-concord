"""Capture the local UI with installed Edge and standard-library CDP."""
import asyncio
import argparse
import base64
import json
import os
import struct
import subprocess
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from uuid import uuid4

class LocalWebSocket:
    """Minimal RFC 6455 client for the local browser debugging endpoint."""
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    @classmethod
    async def connect(cls, url):
        endpoint = urlsplit(url)
        assert endpoint.hostname in {"127.0.0.1", "localhost"}
        reader, writer = await asyncio.open_connection(endpoint.hostname, endpoint.port)
        key = base64.b64encode(os.urandom(16)).decode()
        writer.write((f"GET {endpoint.path} HTTP/1.1\r\nHost: {endpoint.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        await writer.drain()
        response = await reader.readuntil(b"\r\n\r\n")
        assert response.startswith(b"HTTP/1.1 101 "), response
        return cls(reader, writer)

    async def write_message(self, message, opcode=1):
        payload = message.encode() if isinstance(message, str) else message
        size, mask = len(payload), os.urandom(4)
        header = bytes([0x80 | opcode])
        header += bytes([0x80 | size]) if size < 126 else bytes([0xFE]) + struct.pack("!H", size) if size < 65536 else bytes([0xFF]) + struct.pack("!Q", size)
        self.writer.write(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))
        await self.writer.drain()

    async def read_message(self):
        chunks = []
        while True:
            first, second = await self.reader.readexactly(2)
            size = second & 127
            if size == 126:
                size = struct.unpack("!H", await self.reader.readexactly(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", await self.reader.readexactly(8))[0]
            mask = await self.reader.readexactly(4) if second & 128 else None
            payload = await self.reader.readexactly(size)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if first & 15 == 9:
                await self.write_message(payload, 10)
                continue
            if first & 15 == 8:
                raise ConnectionError("Browser closed the debugging socket")
            chunks.append(payload)
            if first & 128:
                return b"".join(chunks).decode()

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/ui-screenshots-round2"
EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


async def main(url, widths=(1440, 1024, 390)):
    if len(widths) > 1:
        results = []
        for width in widths:
            results.extend(await main(url, (width,)))
        (OUT / "layout-check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results
    OUT.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "artifacts" / ("edge-visual-" + uuid4().hex)
    browser = subprocess.Popen([str(EDGE), "--headless=new", "--enable-unsafe-swiftshader", "--no-first-run",
                                "--no-default-browser-check", "--remote-debugging-port=0",
                                "--user-data-dir=" + str(profile), "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for _ in range(40):
            try:
                port = int((profile / "DevToolsActivePort").read_text().splitlines()[0])
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                    targets = json.load(response)
                target = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(.25)
        else:
            raise TimeoutError("Edge did not expose its local debugging endpoint")
        ws = await LocalWebSocket.connect(target["webSocketDebuggerUrl"])
        next_id = 0

        async def call(method, params=None):
            nonlocal next_id
            next_id += 1
            await ws.write_message(json.dumps({"id": next_id, "method": method, "params": params or {}}))
            while True:
                message = await asyncio.wait_for(ws.read_message(), timeout=45)
                result = json.loads(message)
                if result.get("id") == next_id:
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    return result.get("result", {})

        async def evaluate(expression):
            result = await call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
            if "exceptionDetails" in result:
                raise RuntimeError(result["exceptionDetails"])
            return result["result"].get("value")

        async def ready(selector):
            for _ in range(120):
                if await evaluate(f"!!document.querySelector({json.dumps(selector)})"):
                    await asyncio.sleep(1.5)
                    return
                await asyncio.sleep(.25)
            raise TimeoutError(selector)

        async def mobile_sidebar(width):
            if width < 640:
                await evaluate("document.querySelector('[data-testid=stSidebarCollapseButton] button')?.click()")
                await asyncio.sleep(.5)

        async def shot(name):
            result = await call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            (OUT / (name + ".png")).write_bytes(base64.b64decode(result["data"]))

        await call("Page.enable")
        await call("Runtime.enable")
        findings = []
        for width in widths:
            await call("Emulation.setDeviceMetricsOverride", {"width": width, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
            await call("Page.navigate", {"url": url.rstrip("/") + "/?empty=1"})
            await ready(".empty-state")
            await mobile_sidebar(width)
            await shot(f"{width}-empty")
            await call("Page.navigate", {"url": url.rstrip("/") + "/"})
            await ready(".hero-card")
            await ready(".concord-footer")
            await mobile_sidebar(width)
            # Exercise the actual frontend tabs without executing a new investigation.
            await evaluate("[...document.querySelectorAll('[role=tab]')].find(x=>x.textContent.includes('Ask a question'))?.click()")
            await ready(".st-key-user_location")
            await evaluate("[...document.querySelectorAll('[role=tab]')].find(x=>x.textContent.includes('Guided questions'))?.click()")
            await evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0,0)")
            await asyncio.sleep(.4)
            await shot(f"{width}-overview")
            for section, selector in [("hero", ".hero-card"), ("tiles", "#significance"), ("chart", "#time-series"),
                                      ("coupling", "#coupling"), ("simulator", "#simulator"), ("telemetry", "#telemetry")]:
                await evaluate(f"document.querySelector({json.dumps(selector)}).scrollIntoView({{block:'start',behavior:'instant'}})")
                await asyncio.sleep(.7)
                await shot(f"{width}-{section}")
            findings.append(await evaluate("""JSON.parse(JSON.stringify({
                width:innerWidth, bodyWidth:document.body.scrollWidth,
                exceptions:[...document.querySelectorAll('[data-testid=stException]')].map(x=>x.textContent),
                grid:getComputedStyle(document.querySelector('.statistics-grid')).gridTemplateColumns,
                heroFont:getComputedStyle(document.querySelector('.hero-value')).fontFamily,
                bodyFont:getComputedStyle(document.querySelector('.context-stamp')).fontFamily,
                loadButton:[...document.querySelectorAll('[class*=st-key-load_watch_] button')].map(x=>({text:x.textContent,width:x.getBoundingClientRect().width,height:x.getBoundingClientRect().height})),
                vectorImages:[...document.querySelectorAll('img[src^="data:image/svg"]')].length,
                brokenVectors:[...document.querySelectorAll('img[src^="data:image/svg"]')].filter(x=>!x.complete || !x.naturalWidth).length,
                captionStyle:[...document.querySelectorAll('p')].filter(x=>x.textContent.startsWith('Analysis uses')).map(x=>({color:getComputedStyle(x).color,ancestors:[x.parentElement,x.parentElement.parentElement,x.parentElement.parentElement.parentElement].map(p=>({testid:p.dataset.testid,class:p.className,color:getComputedStyle(p).color,opacity:getComputedStyle(p).opacity}))})),
                clipped:[...document.querySelectorAll('.var-card,.hero-card,.stat-tile,.led-tile,[data-testid=stPlotlyChart]')].filter(x=>{let r=x.getBoundingClientRect();return r.right>innerWidth+2 || r.left < -2}).map(x=>x.className),
                headings:[...document.querySelectorAll('.section-heading h2')].map(x=>x.textContent)
            }))"""))
            print(f"Captured {width}px", flush=True)
        (OUT / "layout-check.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
        print(json.dumps(findings, indent=2))
        await call("Browser.close")
        return findings
    finally:
        if browser.poll() is None:
            browser.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    asyncio.run(main(parser.parse_args().url))
