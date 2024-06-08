from dagster import Definitions, load_assets_from_modules, AssetSelection, ScheduleDefinition, define_asset_job

from . import assets

all_assets = load_assets_from_modules([assets])

# defs = Definitions(
#     assets=all_assets,
# )

# Define a job that will materialize the assets
hackernews_job = define_asset_job("hackernews_job", selection=AssetSelection.all())

# Addition: a ScheduleDefinition the job it should run and a cron schedule of how frequently to run it
hackernews_schedule = ScheduleDefinition(
    job=hackernews_job,
    cron_schedule="0 0 */14 * *", 
)

defs = Definitions(
    assets=all_assets,
    schedules=[hackernews_schedule],
)