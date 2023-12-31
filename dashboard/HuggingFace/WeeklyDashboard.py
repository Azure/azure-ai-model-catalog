import os,sys
import requests
import pandas 
import csv
from datetime import datetime
from github import Github, Auth

 

class Dashboard():
    def __init__(self): 
        self.github_token = os.environ['token']
        #self.github_token = "API_TOKEN"
        self.token = Auth.Token(self.github_token)
        self.auth = Github(auth=self.token)
        self.repo = self.auth.get_repo("Azure/azure-ai-model-catalog")
        self.repo_full_name = self.repo.full_name
        self.data = {
            "workflow_id": [], "workflow_name": [], "last_runid": [], "created_at": [],
            "updated_at": [], "status": [], "conclusion": [], "jobs_url": []
        }
        self.models_data = []  # Initialize models_data as an empty list
     
    def get_workflow_names_from_github(self):
        # Fetch the content of your CSV file from your GitHub repository
        file_path = "tests/config/modellist.csv"  # Update this with the actual path
        try:
            url = f"https://raw.githubusercontent.com/{self.repo_full_name}/master/{file_path}"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the CSV content and return it as a list
            csv_data = response.text.splitlines()
            csv_reader = csv.reader(csv_data)
            
            # Assuming the first column contains the data you want to retrieve
            mlflow_prefixed_data = ["MLFlow-" + row[0] for row in csv_reader]
            
            return mlflow_prefixed_data
            
        except Exception as e:
            print(f"Error fetching or parsing content from GitHub: {e}")
            return [] 
         
    def workflow_last_run(self): 
        workflows_to_include = self.get_workflow_names_from_github()
        normalized_workflows = [workflow_name.replace("/","-") for workflow_name in workflows_to_include]
        # normalized_workflows = [hf_name for hf_name in workflows_to_include]
        # hf_name = [hf_name for hf_name in workflows_to_include]
        #print(workflow_name)
        # print(hf_name)
        for workflow_name in normalized_workflows:
            try:
                
                    workflow_runs_url = f"https://api.github.com/repos/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/runs"
                    response = requests.get(workflow_runs_url, headers={"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github.v3+json"})
                    response.raise_for_status()
                    runs_data = response.json()
    
     
    
                    if "workflow_runs" not in runs_data:
                        print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                        continue
    
     
    
                    workflow_runs = runs_data["workflow_runs"]
                    if not workflow_runs:
                        print(f"No runs found for workflow '{workflow_name}'. Skipping...")
                        continue
    
     
    
                    last_run = workflow_runs[0]
                    jobs_response = requests.get(last_run["jobs_url"], headers={"Authorization": f"Bearer {self.github_token}", "Accept": "application/vnd.github.v3+json"})
                    jobs_data = jobs_response.json()
    
     
    
                   # badge_url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml/badge.svg"
                    html_url = jobs_data["jobs"][0]["html_url"] if jobs_data.get("jobs") else ""
    
                    last_run_timestamp = datetime.strptime(last_run["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                    if last_run_timestamp >= datetime(2023, 9, 29):
    
                        self.data["workflow_id"].append(last_run["workflow_id"])
                        self.data["workflow_name"].append(workflow_name.replace(".yml", ""))
                        self.data["last_runid"].append(last_run["id"])
                        self.data["created_at"].append(last_run["created_at"])
                        self.data["updated_at"].append(last_run["updated_at"])
                        self.data["status"].append(last_run["status"])
                        self.data["conclusion"].append(last_run["conclusion"])
                        self.data["jobs_url"].append(html_url)
        
         
        
                        #if html_url:
                            #self.data["badge"].append(f"[![{workflow_name}]({badge_url})]({html_url})")
                        #else:
                            #url = f"https://github.com/{self.repo_full_name}/actions/workflows/{workflow_name}.yml"
                            #self.data["badge"].append(f"[![{workflow_name}]({badge_url})]({url})")
                        run_link = f"https://github.com/{self.repo_full_name}/actions/runs/{last_run['id']}"
                        models_entry = {
                            "Model": workflow_name.replace("MLFlow-",""),
                            # "HFLink": f"[Link](https://huggingface.co/{workflow_name.replace(".yml", "").replace("MLFlow-","")})",
                            # "Status": "<span style='background-color: #00FF00; padding: 2px 6px; border-radius: 3px;'>PASS</span>" if last_run["conclusion"] == "success" else "<span style='background-color: #FF0000; padding: 2px 6px; border-radius: 3px;'>FAIL</span>",
                            # "Status": " ✅ PASS" if last_run["conclusion"] == "success" elif last_run["conclusion"] == "failure" "❌ FAIL",
                            "Status": f"{'✅ PASS' if last_run['conclusion'] == 'success' else '❌ FAIL' if last_run['conclusion'] == 'failure' else '🚫 CANCELLED' if last_run['conclusion'] == 'cancelled' else '⏳ RUNNING'}",
                            "LastRunLink": f"[Link]({run_link})",
                            "LastRunTimestamp": last_run["created_at"]
                        }
        
                        self.models_data.append(models_entry)

 

            except requests.exceptions.RequestException as e:
                print(f"An error occurred while fetching run information for workflow '{workflow_name}': {e}")

 
        # self.models_data.sort(key=lambda x: x["Status"])
        self.models_data.sort(key=lambda x: (x["Status"] != "❌ FAIL", x["Status"]))
        return self.data

    def results(self, last_runs_dict):
        results_dict = {"total": 0, "success": 0, "failure": 0, "cancelled": 0,"running":0, "not_tested": 0, "total_duration": 0}
        summary = []

 

        df = pandas.DataFrame.from_dict(last_runs_dict)
        # df = df.sort_values(by=['status'], ascending=['failure' in df['status'].values])
        results_dict["total"] = df["workflow_id"].count()
        results_dict["success"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'success')]['workflow_id'].count()
        results_dict["failure"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'failure')]['workflow_id'].count()
        results_dict["cancelled"] = df.loc[(df['status'] == 'completed') & (df['conclusion'] == 'cancelled')]['workflow_id'].count()
        results_dict["running"] = df.loc[df['status'] == 'in_progress']['workflow_id'].count()  # Add running count


        success_rate = results_dict["success"]/results_dict["total"]*100.00
        failure_rate = results_dict["failure"]/results_dict["total"]*100.00
        cancel_rate = results_dict["cancelled"]/results_dict["total"]*100.00
        running_rate = results_dict["running"] / results_dict["total"] * 100.00  # Calculate running rate

 

        summary.append("🚀Total|✅Success|❌Failure|🚫Cancelled|⏳Running|")
        summary.append("-----|-------|-------|-------|-------|")
        summary.append(f"{results_dict['total']}|{results_dict['success']}|{results_dict['failure']}|{results_dict['cancelled']}|{results_dict['running']}|")
        summary.append(f"100.0%|{success_rate:.2f}%|{failure_rate:.2f}%|{cancel_rate:.2f}%|{running_rate:.2f}%|")

 

        models_df = pandas.DataFrame.from_dict(self.models_data)
        models_md = models_df.to_markdown()

 

        summary_text = "\n".join(summary)
        current_date = datetime.now().strftime('%Y%m%d')

        with open("ReadmeLatest.md", "w", encoding="utf-8") as f:
            f.write(summary_text)
            f.write(os.linesep)
            f.write(os.linesep)
            f.write(models_md)

 

def main():

        my_class = Dashboard()
        last_runs_dict = my_class.workflow_last_run()
        my_class.results(last_runs_dict)

if __name__ == "__main__":
    main()
