from monailabel.interfaces.app import MONAILabelApp
import os

class MyCustomApp(MONAILabelApp):
    def __init__(self):
        super().__init__(
            app_dir=os.path.dirname(__file__),
            studies=os.path.join(os.path.dirname(__file__), "datasets"),
        )

# Entry point for MONAI Label
def main():
    MyCustomApp().run()

if __name__ == "__main__":
    main()
