"""Optional framework integrations for arcstore's torch layer.

Each submodule pulls in a heavyweight optional dependency (accelerate, ...) and
is imported explicitly by the caller, never eagerly — so ``import arcstore`` and
``import arcstore.torch`` stay free of these extras.
"""
