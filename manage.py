import asyncio

import click
import uvicorn


@click.group()
def cli() -> None:
    pass


@cli.command("start-api", short_help="Запуск FastAPI приложения")
@click.option("--host", default="0.0.0.0", show_default=True)  # noqa: S104
@click.option("--port", default=8000, show_default=True, type=int)
def start_api(host: str, port: int) -> None:
    from rag.adapters.inbound.api.builder import AppBuilder
    from rag.di import container

    async def run() -> None:
        async with container:
            app_builder = await container.get(AppBuilder)
            config = uvicorn.Config(app_builder.create_app(), host=host, port=port, reload=False)
            server = uvicorn.Server(config)
            await server.serve()

    asyncio.run(run())


@cli.command("start-worker", short_help="Запуск Celery-воркера индексации")
@click.option("--concurrency", default=2, show_default=True, type=int)
def start_worker(concurrency: int) -> None:
    from rag.adapters.inbound.worker.app import celery_app
    from rag.adapters.outbound.tasks.celery_app import INDEXING_QUEUE
    from rag.di import settings

    celery_app.worker_main(
        ["worker", "-Q", INDEXING_QUEUE, f"--concurrency={concurrency}", f"--loglevel={settings.log_level}"],
    )


if __name__ == "__main__":
    cli()
