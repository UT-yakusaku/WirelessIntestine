import asyncio
from bleak import BleakClient
import datetime
import struct
import os
from dotenv import load_dotenv

load_dotenv() 
mac_address = os.getenv('MAC_ADDRESS')

def notification_handler(sender, data: bytearray):
    print(struct.unpack('>H', data[0:2]))

async def run(address, loop):
    async with BleakClient(address, loop=loop) as client:
        x = client.is_connected
        print("Connected: {0}".format(x))
        await client.start_notify("6e400003-b5a3-f393-e0a9-e50e24dcca9e", notification_handler)

        while True:
            await asyncio.sleep(100) # 100 -> 0

loop = asyncio.get_event_loop()
loop.run_until_complete(run(mac_address, loop)) #23:01