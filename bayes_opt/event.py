class Events:
    OPTIMIZATION_START = 'optimization:start'
    OPTIMIZATION_STEP = 'optimization:step'
    OPTIMIZATION_END = 'optimization:end'


DEFAULT_EVENTS = [
    Events.OPTIMIZATION_START,
    Events.OPTIMIZATION_STEP,
    Events.OPTIMIZATION_END,
]
