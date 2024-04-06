import numpy as np
import pandas as pd


class client():
    def __init__(self,cid,net,trainset,testset):
        self.cid=cid 
        self.model=net
        self.trainset=trainset
        self.testset=testset

    def get_parameter(self,cid,net,trainset,testset):
        return self.model.get_parameter()
    
    def fit(self, parameters , config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        #print command to know that the config of training process
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        self.model.load_parameter(parameters)
        parameters_prime = self.model.get_parameter()
        num_examples_train = len(self.trainset[0])
        history=self.model.fit()
        results = {
            "D_loss": history["D_loss"][-1],
            "G_loss": history["G_loss"][-1],
        }
        return parameters_prime, num_examples_train, results
    def evaluate(self, parameters, config):
        # model gan đôi khi không cần evaluate lại
        """Evaluate parameters on the locally held test set."""
        print(f"[Client {self.cid}] evaluate, config: {config}")
        # Update local model with global parameters
        self.model.load_parameter(parameters)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.testset[0], self.testset[1])
        num_examples_test = len(self.testset[0])
        return loss, num_examples_test, {"accuracy": accuracy}
    
def client_fn(cid: str) -> client:  
    """Create a Flower client representing a single organization."""

    # Load model
    net = CGAN(0.00)
    print(cid)
    trainset = trainlist[int(cid)]
    testset = testlist[int(cid)]

    # Create a  single Flower client representing a single organization
    return client(cid, net, trainset, testset)


def get_evaluate_fn(model, weight_storage):
    """Return an evaluation function for server-side evaluation."""
    # x_val, y_val = vallist[0]
    y_test=testdata_list[1]
    X_test=testdata_list[0]
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: dict,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.load_parameter(parameters)  # Chuyển đổi thành từ điển và cập nhật model với các thông số mới nhất
        weight_storage.append(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, {"accuracy": accuracy}
    return evaluate


def get_evaluate_fn_plot(model, y_plot):
    """Return an evaluation function for server-side evaluation."""
    global testdata_list
    y_test=testdata_list[1]
    X_test=testdata_list[0]
    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: dict,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.load_parameter(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(X_test, y_test)
        if server_round == NUM_ROUNDS:
          y_plot.append(accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate

class server()-> client:
    def __init__():
        server_model=Net()
        trainset
        testset
        clients=[]
        for i in range(n_clients):
            client=client_fn(i)
            clients.append(client)
        
        







