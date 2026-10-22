"""EMR synthetic backbone variants and names, without model/data imports."""
DEFAULT_HIDDEN_CHANNELS = 64


def validate_hidden_channels(value):
    width = DEFAULT_HIDDEN_CHANNELS if value is None else int(value)
    if width < 32 or width % 32:
        raise ValueError("EMR-Diff --hidden_channels must be a positive multiple of 32 (e.g. 64 or 128)")
    return width


def variant_suffix(width=None):
    width = validate_hidden_channels(width)
    return "" if width == DEFAULT_HIDDEN_CHANNELS else f"_base{width}"


def checkpoint_name(kind, width=None):
    if kind not in ("best", "last"):
        raise ValueError("EMR training keeps only best and last checkpoints")
    return f"{kind}{variant_suffix(width)}.pth.tar"


def architecture_id(width=None):
    return f"fixed{validate_hidden_channels(width)}_dynamic_io_v2"


def checkpoint_width(checkpoint, requested=None):
    width = validate_hidden_channels(checkpoint.get("backbone_width", DEFAULT_HIDDEN_CHANNELS))
    if requested is not None and validate_hidden_channels(requested) != width:
        raise ValueError(f"EMR checkpoint backbone_width={width} differs from requested hidden_channels={requested}")
    return width


def add_hidden_channels_argument(parser):
    parser.add_argument("--hidden_channels", type=validate_hidden_channels, default=None,
                        help="EMR-Diff backbone width; default=64, e.g. 128 uses *_base128.pth.tar")
