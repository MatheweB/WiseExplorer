import os
import json
from peewee import (
    Model,
    IntegerField,
    BlobField,
    TextField,
    SqliteDatabase,
    CompositeKey,
    DoesNotExist,
)

db = SqliteDatabase(None)


class Position(Model):
    """One row per unique board."""
    game_id = TextField()
    board_hash = IntegerField(primary_key=True)
    board_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        database = db
        table_name = "positions"


class Move(Model):
    """One row per unique move."""
    game_id = TextField()
    move_hash = IntegerField(primary_key=True)
    move_bytes = BlobField()
    meta_json = TextField()

    class Meta:
        database = db
        table_name = "moves"


class PlayStats(Model):
    """Outcome counts only."""
    game_id = TextField()
    board_hash = IntegerField()
    move_hash = IntegerField()
    snapshot_player = IntegerField()
    acting_player = IntegerField()

    win_count = IntegerField(default=0)
    tie_count = IntegerField(default=0)
    loss_count = IntegerField(default=0)

    class Meta:
        database = db
        table_name = "play_stats"
        primary_key = CompositeKey(
            "game_id", "board_hash", "move_hash", "snapshot_player", "acting_player"
        )
