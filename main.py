"""Entry point for the ICH multi-task segmentation + classification pipeline."""

from src import Config, apply_overrides, build_parser, set_seed, setup_logging


def main() -> None:
    """Parse CLI args, apply config overrides, then dispatch the chosen subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config()
    cfg = apply_overrides(cfg, args)

    set_seed(cfg.seed)
    cfg.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    logger.info(
        "Running '%s' on device=%s with output=%s",
        args.command,
        cfg.device,
        cfg.output,
    )

    args.func(cfg, logger, args)


if __name__ == "__main__":
    main()
