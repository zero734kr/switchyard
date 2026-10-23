"""Allow `python -m switchyard` as well as the `switchyard` console script."""

from .cli import main

raise SystemExit(main())
