class NewEvaluator:
    def __init__(self, scheme):
        self.scheme = scheme
        self.backend = scheme.backend

    def __del__(self):
        self.backend.DeleteBootstrappers()

    def generate_bootstrapper(self, slots, level=None):
        # We will wait to instantiate any bootstrapper until our bootstrap
        # placement algorithm determines they're necessary. A level below the
        # scheme's maximum builds a bootstrapper that refreshes to that level.
        logp = self.scheme.params.get_boot_logp()
        if level is None:
            return self.backend.NewBootstrapper(logp, slots)
        return self.backend.NewBootstrapperAtLevel(logp, slots, int(level))
    
    def bootstrap(self, ctxt, slots, level=None):
        if level is None:
            return self.backend.Bootstrap(ctxt, slots)
        return self.backend.BootstrapAtLevel(ctxt, slots, int(level))