#!/bin/sh
set -e

echo "Waiting for database and ensuring target database exists..."
python3 - << 'EOF'
import asyncio, os, asyncpg, re

async def main():
    host = os.getenv('POSTGRES_HOST', 'postgres')
    port = int(os.getenv('POSTGRES_PORT', '5432'))
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'postgres')
    target_db = os.getenv('POSTGRES_DB', 'app_db')

    for i in range(30):
        try:
            conn = await asyncpg.connect(
                host=host, port=port, user=user, password=password, database='postgres'
            )
            db_exists = await conn.fetchval(
                'SELECT 1 FROM pg_database WHERE datname = $1', target_db
            )
            if not db_exists:
                safe_db_name = re.sub(r'[^a-zA-Z0-9_]', '', target_db)
                await conn.execute(f'CREATE DATABASE "{safe_db_name}"')
                print(f'Database "{safe_db_name}" created successfully!')
            else:
                print(f'Database "{target_db}" is ready!')
            await conn.close()
            return
        except Exception as e:
            print(f'Waiting for database ({host}:{port})... ({i+1}/30): {e}')
            await asyncio.sleep(1)
    raise RuntimeError('Could not connect to database after 30 attempts')

asyncio.run(main())
EOF

echo "Running database migrations..."
alembic upgrade head

echo "Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

