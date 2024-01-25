from utils.logging import get_logger
from fetch_task import HfTask
import re

logger = get_logger(__name__)

class ModelDetail:
    def __init__(self, workspace_ml_client) -> None:
        self.workspace_ml_client = workspace_ml_client

    def get_latest_model_version(self, model_name):
        logger.info("In get_latest_model_version...")
        version_list = list(self.workspace_ml_client.models.list(model_name))
        if len(version_list) == 0:
            logger.info("Model not found in registry")
        else:
            #model_version = version_list[0].version
            model_version = 10
            foundation_model = self.workspace_ml_client.models.get(
                model_name, model_version)
            logger.info(
                "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
                    foundation_model.name, foundation_model.version, foundation_model.id
                )
            )
        logger.info(
            f"Latest model {foundation_model.name} version {foundation_model.version} created at {foundation_model.creation_context.created_at}")
        #print(f"Model Config : {latest_model.config}")
        return foundation_model
        
    def get_model_detail(self, test_model_name):
        expression_to_ignore = ["/", "\\", "|", "@", "#", ".",
                                "$", "%", "^", "&", "*", "<", ">", "?", "!", "~"]
        # Create the regular expression to ignore
        regx_for_expression = re.compile(
            '|'.join(map(re.escape, expression_to_ignore)))
        # Check the model_name contains any of there character
        expression_check = re.findall(
            regx_for_expression, test_model_name)
        if expression_check:
            # Replace the expression with hyphen
            model_name = regx_for_expression.sub("-", test_model_name)
        else:
            model_name = test_model_name
        latest_model = self.get_latest_model_version(model_name)
        # try:
        #     task = latest_model.flavors["transformers"]["task"]
        # except Exception as e:
        #     logger.warning(
        #         f"::warning::From the transformer flavour we are not able to extract the task for this model : {latest_model}")
        #     logger.info(f"Following Alternate approach to getch task....")
        #     hfApi = HfTask(model_name=test_model_name)
        #     task = hfApi.get_task()
        logger.info(f"latest_model: {latest_model}")
        # logger.info(f"Task is : {task}")
        return latest_model
