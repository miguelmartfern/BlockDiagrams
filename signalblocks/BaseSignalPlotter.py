class BaseSignalPlotter:
    """
    Base class for SignalPlotter and DiscreteSignalPlotter to unify
    shared initialization, parameters, and utilities for consistent
    continuous and discrete signal plotting in SignalBlocks.
    """
    def __init__(
        self,
        horiz_range=(-5, 5),
        vert_range=None,
        period=None,
        figsize=(8, 3),
        tick_size_px=5,
        xticks='auto',
        yticks='auto',
        xtick_labels=None,
        ytick_labels=None,
        xticks_delta=None,
        yticks_delta=None,
        fraction_ticks=False,
        save_path=None,
        show_plot=True,
        color='black',
        alpha=0.5
    ):
        """
        Initialize common parameters for continuous and discrete signal plotters.
        """
        self.signal_defs = {}
        self.var_symbols = {}
        self.custom_labels = {}
        self.signal_periods = {}
        self.current_name = None

        self.horiz_range = horiz_range
        self.vert_range = vert_range
        self.period = period

        self.figsize = figsize
        self.tick_size_px = tick_size_px
        self.color = color
        self.alpha = alpha

        self.save_path = save_path
        self.show_plot = show_plot

        self.fraction_ticks = fraction_ticks

        self.init_xticks_arg = xticks
        self.init_yticks_arg = yticks

        self.xtick_labels = xtick_labels
        self.ytick_labels = ytick_labels

        if xtick_labels is not None and xticks not in [None, 'auto']:
            if len(xtick_labels) != len(xticks):
                raise ValueError("xtick_labels and xticks must have the same length")

        if ytick_labels is not None and yticks not in [None, 'auto']:
            if len(ytick_labels) != len(yticks):
                raise ValueError("ytick_labels and yticks must have the same length")

        self.xticks_delta = xticks_delta
        self.yticks_delta = yticks_delta
