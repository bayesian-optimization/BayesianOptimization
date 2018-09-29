# Inspired/Taken from https://www.protechtraining.com/blog/post/879#simple-observer


class Observer:
    def update(self, event, instance):
        # Avoid circular import
        from .bayesian_optimization import Events
        if event is Events.INIT_DONE:
            print("Initialization completed")
        elif event is Events.FIT_STEP_DONE:
            print("Optimization step finished, current max: ", instance.res['max'])
        elif event is Events.FIT_DONE:
            print("Optimization finished, maximum value at: ", instance.res['max'])


class Observable(object):
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self.events = {event: dict()
                       for event in events}

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