import torchvision
import flwr as fl
import warnings
from going_modular import *

warnings.simplefilter("ignore")

print("flwr", fl.__version__)
print("numpy", np.__version__)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)

main_parser2 = parsing(description='Federated Learning asset')
args = main_parser2.parse_args()
CLASSES = classes_string()

DEVICE = torch.device(choice_device(args.device))
print(f"Training on {DEVICE}")

trainloaders, valloaders, testloader = data_setup.load_datasets(num_clients=args.number_clients,
                                                                batch_size=args.batch_size,
                                                                splitter=args.split)
