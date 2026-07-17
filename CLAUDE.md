This project uses `uv` to manage Python environments and dependencies.
To run python: `uv run python`.
To check format: `uv run ruff check`

Use English for code, comments, docs, commit messages, etc. (things that will be submitted to Git).

After changing the code, do not forget to update the corresponding tests/docs if they exist.

Before committing, review the type of change in this round (patch/bug-fix, improvement, breaking, ...),
and change the SemVer of the project accordingly.
(NOTE: If X == 0, Y becomes the major version number and Z becomes the minor version number.)

Do format checking and testing here.

When git committing, only concisely describe the changes/fixes/improvements.
DO NOT add additional information or signature such as "Co-Authored-By: XXX".

If version is changed, run `git tag -a vX.Y.Z -m "vX.Y.Z"` to add a tag.
