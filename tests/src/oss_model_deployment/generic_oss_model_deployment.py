from azureml.core import Workspace, Environment
from model_inference_and_deployment import ModelInferenceAndDeployemnt
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.core.exceptions import (
    ResourceNotFoundError
)
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint
)
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
import mlflow
import json
import os
import sys
from box import ConfigBox
from utils.logging import get_logger
from fetch_model_detail import ModelDetail
from azure.ai.ml.dsl import pipeline
from fetch_task import HfTask
from dataset_loader import LoadDataset
from metrics_result import MetricsCalaulator
import time
import pandas as pd

# constants
check_override = True

logger = get_logger(__name__)


def get_error_messages():
    # load ../config/errors.json into a dictionary
    with open('../../config/errors.json') as f:
        return json.load(f)


error_messages = get_error_messages()
compute_name = "model-import-cluster"
# model to test
test_model_name = os.environ.get('test_model_name')

actual_model_name = os.environ.get('actual_model_name')

# test cpu or gpu template
test_sku_type = os.environ.get('test_sku_type')

# bool to decide if we want to trigger the next model in the queue
test_trigger_next_model = os.environ.get('test_trigger_next_model')

# test queue name - the queue file contains the list of models to test with with a specific workspace
test_queue = os.environ.get('test_queue')

# test set - the set of queues to test with. a test queue belongs to a test set
test_set = os.environ.get('test_set')

# bool to decide if we want to keep looping through the queue,
# which means that the first model in the queue is triggered again after the last model is tested
test_keep_looping = os.environ.get('test_keep_looping')

# function to load the workspace details from test queue file
# even model we need to test belongs to a queue. the queue name is passed as environment variable test_queue
# the queue file contains the list of models to test with with a specific workspace
# the queue file also contains the details of the workspace, registry, subscription, resource group

huggingface_model_exists_in_registry = False

FILE_NAME = "pipeline_task.json"


def get_test_queue() -> ConfigBox:
    queue_file = f"../../config/queue/{test_set}/{test_queue}.json"
    with open(queue_file) as f:
        return ConfigBox(json.load(f))
# function to load the sku override details from sku-override file
# this is useful if you want to force a specific sku for a model


def get_sku_override():
    try:
        with open(f'../../config/sku-override/{test_set}.json') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"::warning:: Could not find sku-override file: \n{e}")
        return None


# finds the next model in the queue and sends it to github step output
# so that the next step in this job can pick it up and trigger the next model using 'gh workflow run' cli command
def set_next_trigger_model(queue):
    logger.info("In set_next_trigger_model...")
# file the index of test_model_name in models list queue dictionary
    model_list = list(queue.models)
    #model_name_without_slash = test_model_name.replace('/', '-')
    check_oss_base_model = "oss-deployment-"+test_model_name

    if check_oss_base_model in model_list:
        index = model_list.index(check_oss_base_model)
    else:
        index = model_list.index(
            test_model_name.replace("/", "-").lower()+"-oss")

    logger.info(f"index of {test_model_name} in queue: {index}")
# if index is not the last element in the list, get the next element in the list
    if index < len(model_list) - 1:
        next_model = model_list[index + 1]
    else:
        if (test_keep_looping == "true"):
            next_model = queue[0]
        else:
            logger.warning("::warning:: finishing the queue")
            next_model = ""
# write the next model to github step output
    with open(os.environ['GITHUB_OUTPUT'], 'a') as fh:
        logger.info(f'NEXT_MODEL={next_model}')
        print(f'NEXT_MODEL={next_model}', file=fh)


def create_or_get_compute_target(ml_client, compute, instance_type):
    cpu_compute_target = compute
    try:
        compute = ml_client.compute.get(cpu_compute_target)
    except ResourceNotFoundError:
        logger.info("Creating a new compute...")
        compute = AmlCompute(
            name=cpu_compute_target, size=instance_type, idle_time_before_scale_down=120, min_instances=0, max_instances=4
        )
        ml_client.compute.begin_create_or_update(compute).result()

    return compute


def get_file_path(task):
    file_name = task+".json"
    data_path = f"./datasets/{task}/{file_name}"
    return data_path


def get_dataset(task, data_path, latest_model):
    load_dataset = LoadDataset(
        task=task, data_path=data_path, latest_model=latest_model)
    task = task.replace("-", "_")
    attribute = getattr(LoadDataset, task)
    return attribute(load_dataset)


def get_pipeline_task(task):
    try:
        with open(FILE_NAME) as f:
            pipeline_task = ConfigBox(json.load(f))
            logger.info(
                f"Library name based on its task :\n\n {pipeline_task}\n\n")
    except Exception as e:
        logger.error(
            f"::Error:: Could not find library from here :{pipeline_task}.Here is the exception\n{e}")
    return pipeline_task.get(task)


def create_endpoint(workspace_ml_client, endpoint_name):
    try:
        logger.info("Inside creating the endpoint method")
        endpoint = workspace_ml_client.online_endpoints.get(name=endpoint_name)
        return endpoint
    except ResourceNotFoundError as e:
        logger.warning("The endpoint do not exist and now creting new endpoint")
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key"
        )
        logger.warning("update the endpoint in the workspace")
        workspace_ml_client.online_endpoints.begin_create_or_update(
            endpoint).wait()
        return endpoint
    except Exception as e:
        logger.error(f"Failed due to this : {e}")
        sys.exit(1)
# @pipeline
# def model_import_pipeline(compute_name, update_existing_model, task_name):
#     import_model = registry_ml_client.components.get(
#         name="import_model_oss_test", label="latest")
#     import_model_job = import_model(model_id=test_model_name, compute=compute_name,
#                                     task_name=task_name, update_existing_model=update_existing_model)
#     # Set job to not continue on failure
#     import_model_job.settings.continue_on_step_failure = False
#     return {"model_registration_details": import_model_job.outputs.model_registration_details}


# @pipeline()
# def evaluation_pipeline(task, mlflow_model, test_data, input_column_names, label_column_name, evaluation_file_path, compute):
#     try:
#         logger.info("Started configuring the job")
#         #data_path = "./datasets/translation.json"
#         pipeline_component_func = registry_ml_client.components.get(
#             name="mlflow_oss_model_evaluation_pipeline", label="latest"
#         )
#         evaluation_job = pipeline_component_func(
#             # specify the foundation model available in the azureml system registry or a model from the workspace
#             # mlflow_model = Input(type=AssetTypes.MLFLOW_MODEL, path=f"{mlflow_model_path}"),
#             mlflow_model=mlflow_model,
#             # test data
#             test_data=Input(type=AssetTypes.URI_FILE, path=test_data),
#             # The following parameters map to the dataset fields
#             input_column_names=input_column_names,
#             label_column_name=label_column_name,
#             # compute settings
#             compute_name=compute,
#             # specify the instance type for serverless job
#             # instance_type= "STANDARD_NC24",
#             # Evaluation settings
#             task=task,
#             # config file containing the details of evaluation metrics to calculate
#             # evaluation_config=Input(
#             #     type=AssetTypes.URI_FILE, path="./evaluation/eval_config.json"),
#             evaluation_config=Input(
#                 type=AssetTypes.URI_FILE, path=evaluation_file_path),
#             # config cluster/device job is running on
#             # set device to GPU/CPU on basis if GPU count was found
#             device="auto",
#         )
#         return {"evaluation_result": evaluation_job.outputs.evaluation_result}
#     except Exception as ex:
#         _, _, exc_tb = sys.exc_info()
#         logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
#                      f" the exception is this one : \n {ex}")
#         raise Exception(ex)


if __name__ == "__main__":
    # if any of the above are not set, exit with error
    if test_model_name is None or test_sku_type is None or test_queue is None or test_set is None or test_trigger_next_model is None or test_keep_looping is None:
        logger.error("::error:: One or more of the environment variables test_model_name, test_sku_type, test_queue, test_set, test_trigger_next_model, test_keep_looping are not set")
        exit(1)

    queue = get_test_queue()

    # sku_override = get_sku_override()
    # if sku_override is None:
    #     check_override = False

    # if test_trigger_next_model == "true":
    #     set_next_trigger_model(queue)
    # print values of all above variables
    logger.info(f"test_subscription_id: {queue['subscription']}")
    logger.info(f"test_resource_group: {queue['subscription']}")
    logger.info(f"test_workspace_name: {queue['workspace']}")
    logger.info(f"test_model_name: {test_model_name}")
    logger.info(f"test_sku_type: {test_sku_type}")
    logger.info(f"test_trigger_next_model: {test_trigger_next_model}")
    logger.info(f"test_queue: {test_queue}")
    logger.info(f"test_set: {test_set}")
    logger.info(f"Here is my test model name : {test_model_name}")
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()
    logger.info(f"workspace_name : {queue.workspace}")
    try:
        workspace_ml_client = MLClient.from_config(credential=credential)
    except:
        workspace_ml_client = MLClient(
            credential=credential,
            subscription_id=queue.subscription,
            resource_group_name=queue.resource_group,
            workspace_name=queue.workspace
        )
    ws = Workspace(
        subscription_id=queue.subscription,
        resource_group=queue.resource_group,
        workspace_name=queue.workspace
    )
    # registry_ml_client = MLClient(
    #     credential=credential,
    #     registry_name=queue.registry
    # )
    #
    # mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    registered_model_name = test_model_name.replace("/", "-")
    model_detail = ModelDetail(workspace_ml_client=workspace_ml_client)
    registered_model = model_detail.get_model_detail(
        test_model_name=registered_model_name)

    azureml_registry = MLClient(credential, registry_name="azureml")

    model_detail = ModelDetail(workspace_ml_client=azureml_registry)
    foundation_model = model_detail.get_model_detail(
        test_model_name=test_model_name.replace("/", "-"))
    instance_type = list(foundation_model.properties.get(
        "inference-recommended-sku").split(","))[0]
    compute = instance_type.replace("_", "-")
    logger.info(f"instance : {instance_type} and compute is : {compute}")

    # compute_target = create_or_get_compute_target(
    #                  ml_client=workspace_ml_client,
    #                  compute=compute,
    #                  instance_type=instance_type
    #                  )

    compute_target = create_or_get_compute_target(
        ml_client=workspace_ml_client, compute=compute, instance_type=instance_type)
    #endpoint_name = queue.workspace.split("-")[-1] + "-" + compute.lower()
    endpoint_name = compute.lower()
    endpoint = create_endpoint(
        workspace_ml_client=workspace_ml_client,
        endpoint_name=endpoint_name
    )
    #task = foundation_model.tags["task"]
    task = HfTask(model_name=actual_model_name).get_task(foundation_model=registered_model)
    logger.info(f"Task is this : {task} for the model : {test_model_name}")
    #timestamp = str(int(time.time()))
    #exp_model_name = test_model_name.replace('/', '-')
    # try:
    #     pipeline_object = model_import_pipeline(
    #         compute_name=compute,
    #         task_name=task,
    #         update_existing_model=True,
    #     )
    #     pipeline_object.identity = UserIdentityConfiguration()
    #     pipeline_object.settings.force_rerun = True
    #     pipeline_object.settings.default_compute = compute
    #     schedule_huggingface_model_import = (
    #         not huggingface_model_exists_in_registry
    #         and test_model_name not in [None, "None"]
    #         and len(test_model_name) > 1
    #     )
    #     logger.info(
    #         f"Need to schedule run for importing {test_model_name}: {schedule_huggingface_model_import}")

    #     huggingface_pipeline_job = None
    #     # if schedule_huggingface_model_import:
    #     # submit the pipeline job
    #     huggingface_pipeline_job = workspace_ml_client.jobs.create_or_update(
    #         pipeline_object, experiment_name=f"import-pipeline-{exp_model_name}-{timestamp}"
    #     )
    #     # wait for the pipeline job to complete
    #     workspace_ml_client.jobs.stream(huggingface_pipeline_job.name)
    # except Exception as ex:
    #     _, _, exc_tb = sys.exc_info()
    #     logger.error(f"::error:: Not able to initiate job \n")
    #     logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
    #                  f" skipping the further process and the exception is this one : {ex}")
    #     sys.exit(1)

    # registered_model_detail = ModelDetail(
    #     workspace_ml_client=workspace_ml_client)
    # registered_model = registered_model_detail.get_model_detail(
    #     test_model_name=registered_model_name)

    # try:
    #     flavour = registered_model.flavors
    #     if flavour.get("python_function", None) == None:
    #         logger.info(
    #             f"This model {registered_model.name} is not registered in the mlflow flavour so skipping the further process")
    #         raise Exception(
    #             f"This model {registered_model.name} is not registered in the mlflow flavour so skipping the further process")
    #     else:
    #         if flavour.get("python_function").get("loader_module", None) == "mlflow.transformers":
    #             logger.info(
    #                 f"This model {registered_model.name} is registered in the mlflow flavour")
    #         else:
    #             logger.info(
    #                 f"This model {registered_model.name} is not registered in the mlflow flavour so skipping the further process")
    #             raise Exception(
    #                 f"This model {registered_model.name} is not registered in the mlflow flavour so skipping the further process")
    # except Exception as ex:
    #     _, _, exc_tb = sys.exc_info()
    #     logger.error(f"::error:: Not able to initiate job \n")
    #     logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
    #                  f" skipping the further process and the exception is this one : {ex}")
    #     sys.exit(1)

    # data_path = get_file_path(task=task)
    # input_column_names, label_column_name = get_dataset(task=task, data_path=data_path,
    #                                                     latest_model=registered_model)
    # pieline_task = get_pipeline_task(task)

    # try:
    #     pipeline_jobs = []
    #     eval_experiment_name = f"{pieline_task}-{exp_model_name}-evaluation-{timestamp}"
    #     pipeline_object = evaluation_pipeline(
    #         task=pieline_task,
    #         mlflow_model=Input(type=AssetTypes.MLFLOW_MODEL,
    #                            path=f"{registered_model.id}"),
    #         test_data=Input(type=AssetTypes.URI_FILE, path=data_path),
    #         input_column_names=input_column_names,
    #         label_column_name=label_column_name,
    #         evaluation_file_path=Input(
    #             type=AssetTypes.URI_FILE, path=f"./evaluation/{task}/eval_config.json"),
    #         compute=compute,
    #         #mlflow_model = f"{latest_model.id}",
    #         #data_path = data_path
    #     )
    #     # don't reuse cached results from previous jobs
    #     pipeline_object.settings.force_rerun = True
    #     pipeline_object.settings.default_compute = compute

    #     # set continue on step failure to False
    #     pipeline_object.settings.continue_on_step_failure = False

    #     pipeline_object.display_name = f"eval-{registered_model.name}-{timestamp}"
    #     pipeline_job = workspace_ml_client.jobs.create_or_update(
    #         pipeline_object, experiment_name=eval_experiment_name
    #     )
    #     # add model['name'] and pipeline_job.name as key value pairs to a dictionary
    #     pipeline_jobs.append(
    #         {"model_name": registered_model.name, "job_name": pipeline_job.name})
    #     # wait for the pipeline job to complete
    #     workspace_ml_client.jobs.stream(pipeline_job.name)
    #     # return pipeline_jobs
    #     metrics_df = MetricsCalaulator(
    #         pipeline_jobs=pipeline_jobs, mlflow=mlflow, experiment_name=eval_experiment_name).display_metric()
    #     logger.info(f"Evaluation result is this : {metrics_df}")
    # except Exception as ex:
    #     _, _, exc_tb = sys.exc_info()
    #     logger.error(f"The exception occured at this line no : {exc_tb.tb_lineno}" +
    #                  f" the exception is this one : \n {ex}")
    #     raise Exception(ex)
    logger.info("Proceeding with inference and deployment")
    InferenceAndDeployment = ModelInferenceAndDeployemnt(
        test_model_name=registered_model_name.lower(),
        workspace_ml_client=workspace_ml_client
    )
    InferenceAndDeployment.model_infernce_and_deployment(
        instance_type=instance_type,
        task=task,
        latest_model=registered_model,
        compute=compute,
        endpoint=endpoint,
        actual_model_name=actual_model_name
    )
