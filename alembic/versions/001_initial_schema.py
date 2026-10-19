"""initial_schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-06 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=10), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=False),
        sa.Column("chunking_strategy", sa.String(length=50), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_file_hash", "documents", ["file_hash"], unique=False)
    op.create_index(
        "idx_doc_hash_strategy",
        "documents",
        ["file_hash", "chunking_strategy"],
        unique=False,
    )

    op.create_table(
        "bookings",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("booking_date", sa.String(length=10), nullable=False),
        sa.Column("booking_time", sa.String(length=10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_bookings_session_id", "bookings", ["session_id"], unique=False)
    op.create_index(
        "idx_booking_date_time",
        "bookings",
        ["booking_date", "booking_time"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_booking_date_time", table_name="bookings")
    op.drop_index("ix_bookings_session_id", table_name="bookings")
    op.drop_table("bookings")
    op.drop_index("idx_doc_hash_strategy", table_name="documents")
    op.drop_index("ix_documents_file_hash", table_name="documents")
    op.drop_table("documents")
