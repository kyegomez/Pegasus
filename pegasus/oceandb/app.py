import pegasus.oceandb
import pegasus.oceandb.config
import pegasus.oceandb.server.fastapi import FastAPI

settings = oceandb.config.Settings()
server = FastAPI(settings)
app = server.app()
