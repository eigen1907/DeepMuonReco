from lightning.pytorch.callbacks import Callback
from datetime import datetime as dt


class TimePrinter(Callback):

    def __init__(self):
        super().__init__()

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🚀 on_sanity_check_start')

    def on_sanity_check_end(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🏁 on_sanity_check_end')

    def on_fit_start(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🚀 on_fit_start')

    def on_fit_end(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🏁 on_fit_end')

    def on_test_start(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🚀 on_test_start')

    def on_test_end(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🏁 on_test_end')

    def on_predict_start(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🚀 on_predict_start')

    def on_predict_end(self, trainer, pl_module) -> None:
        print(f'[{dt.now()}] 🏁 on_predict_end')
