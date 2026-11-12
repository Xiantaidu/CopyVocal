class Vocal2MidiError(Exception):
    def __init__(self, message: str = "", *, details: str = ""):
        # Fold details into the message so user-facing tracebacks/dialogs carry them.
        super().__init__(f"{message}: {details}" if details else message)
        self.details = details


class ModelNotFoundError(Vocal2MidiError):
    pass


class ASRError(Vocal2MidiError):
    pass


class AlignmentError(Vocal2MidiError):
    pass


class ExportError(Vocal2MidiError):
    pass


class CancellationError(Vocal2MidiError):
    pass