raise ArtifactValidationError(
    f"chunks.jsonl: missing required field 'chunk_id' on line {line_number}"
)