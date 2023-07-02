class BaseConstraint():
        
        @classmethod
        def create(cls, instance,var):
            setattr(instance, cls._name ,var) if var is not None else setattr(instance, cls._name, list())

        def __lt__(self, other):
            return self._value < other._value