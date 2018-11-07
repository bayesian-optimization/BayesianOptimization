"""


Inspired/Taken from https://www.protechtraining.com/blog/post/879#simple-observer
"""


class Events(object):
    INIT_DONE = 'initialized'
    FIT_STEP_DONE = 'fit_step_done'
    FIT_DONE = 'fit_done'


DEFAULT_EVENTS = [
    Events.INIT_DONE,
    Events.FIT_STEP_DONE,
    Events.FIT_DONE
]


class Observable(object):
    def __init__(self, events=None):
        # maps event names to subscribers
        # str -> dict
        if events is None:
            events = DEFAULT_EVENTS

        self.events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self.events[event]

    def register(self, event, who, callback=None):
        if callback == None:
            callback = getattr(who, 'update')
        self.get_subscribers(event)[who] = callback

    def unregister(self, event, who):
        del self.get_subscribers(event)[who]

    def dispatch(self, event):
        for subscriber, callback in self.get_subscribers(event).items():
            callback(event, self)


class Observer:
    def update(self, event, instance):
        # Avoid circular import
        from .bayesian_optimization import Events
        if event is Events.INIT_DONE:
            print("Initialization completed")
        elif event is Events.FIT_STEP_DONE:
            print("Optimization step finished, current max: ",
                  instance.res['max'])
        elif event is Events.FIT_DONE:
            print("Optimization finished, maximum value at: ",
                  instance.res['max'])
