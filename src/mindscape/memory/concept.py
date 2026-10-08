"""Explicit descriptive concepts; never a numerical oracle or learned evidence."""
from dataclasses import dataclass, field
@dataclass
class ConceptMemory:
    concepts: dict[str,str] = field(default_factory=dict)
    def store(self,name,description):
        if not isinstance(name,str) or not isinstance(description,str):raise TypeError('Concept text required')
        self.concepts[name]=description
    def retrieve(self,name):return self.concepts.get(name)
