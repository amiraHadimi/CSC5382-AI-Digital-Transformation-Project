"""
Feast Feature View definitions for the Agile Story Point Estimation project.
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, Float32

issue_entity = Entity(
    name="issue_id",
    description="Unique identifier for a JIRA issue / user story.",
)

user_story_source = FileSource(
    path=r"C:\Users\dell\Documents\AUI\master SPRING\AI for digital transformation\CSC5382-AI-Digital-Transformation-Project\Milestone 3\data\processed\processed_data.parquet",
    timestamp_field="event_timestamp",
    description="Processed Agile user story features from Milestone 3 pipeline.",
)

user_story_features = FeatureView(
    name="user_story_features",
    entities=[issue_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="text_length", dtype=Int64, description="Character count of input_text"),
        Field(name="word_count", dtype=Int64, description="Word count of input_text"),
        Field(name="has_description", dtype=Int64, description="1 if description is non-empty"),
        Field(name="title_word_count", dtype=Int64, description="Word count of cleaned_title"),
        Field(name="log_storypoints", dtype=Float32, description="log1p of story point target"),
        Field(name="is_fibonacci", dtype=Int64, description="1 if storypoints is Fibonacci"),
    ],
    source=user_story_source,
    description="Engineered features for story point estimation.",
)
