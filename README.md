# Telco Churn Predictions

This repository contains code to build and deploy a machine learning
model that predicts customer churn for a telecommunications company
using the Marimo framework. The model is trained on the well known
IBM Telco Customer Churn dataset.


## The Model

The model is a simple logistic regression model built using `scikit-learn`.

The model currently uses the following features:
- `tenure`: Number of months the customer has stayed with the company.
- `MonthlyCharges`: The amount charged to the customer monthly.
- `TechSupport_yes`: Binary indicator of whether the customer has tech support.

We recommend looking into additional features and engineering new ones to
improve model performance. There are other possible issues to explore, including
model choice, leakage, and others.

The model is built and trained using `notebooks/telco_marimo.py`. The model is
saved in the `models/` directory, with the scaler and model bundled together
using `joblib`.

An example prediction can be found in `prediction.py`.

## CI Pipeline

There is a CI pipeline set up using GitHub Actions that runs tests
on every push and pull request. The tests are located in the `tests/`
directory and can be run locally using `pytest`.

# Deploying to Serverless

The saved model can deployment to an Azure Function with the following
steps:


1. **In your Codespace**, install the Azure Function extension:
    - Open the Extensions view in the left sidebar.
    - Search for "Azure Functions" and install the extension by Microsoft.

2. **In your Codespace**, create the function:
    - Open the Command Palette (`Ctrl+Shift+P`).
    - Type and select *Azure Functions: Create Function*
         - Do not select the "...in Azure..." command
    - Select the root folder of the project (should be the default).
    - Select *Python* as the language.
    - Select *HTTP trigger* as the template.
    - Provide a name (without spaces or special characters) e.g. predict.
    - Select *FUNCTION* as the authorization level.
    - **Do not** *overwrite* the `.gitignore` file.

3. **In your Codespace**, wait some time for the function to be created. Then:
    - Add azure.functions to your env with `uv add azure-functions`
    - Update the `requirements.txt` file for Azure `uv pip freeze > requirements.txt`
    - Edit the `function_app.py` file
        - Add an import: `from prediction import make_prediction`
        - Replace the line `name = req.params.get('name')`,
        using the same approach to get input data for prediction.
            - You'll need tenure, monthly bill and tech support status.
            - It is up to you how you name these but simple names are best.
            - The variable name (on the left) is internal to the function,
            while the string name (on the right) is what the user will need to provide.
                ```python
                tenure = req.params.get('tenure')
                monthly = req.params.get('monthly')
                techsupport = req.params.get('techsupport')
                ```
        - Remove the entire `if not name:` block. We aren't supporting JSON input here.
        - Call the `make_prediction` function, passing tenure, monthly and techsupport
        as keyword arguments following the column names used by the model:
            ```python
            prediction = make_prediction(
                tenure=tenure,
                MonthlyCharges=monthly,
                TechSupport_yes=techsupport
            )
            ```
        - Change `if name:` to `if tenure and monthly and techsupport:`
        - Change the `f""` response to return the prediction result instead of a name.
    - **Commit and Sync** all your changes!

4. **In the Azure Portal**, create a Azure Function App.
    - Choose *Flex Consumption*.
    - Select your existing **Resource Group**.
    - Choose a unique **name** for your Function App.
    - Set the **Region** to a supported student region (e.g. *Switzerland North*).
    - Choose *Python* *3.12* / *3.13* as the **runtime** stack and **version**.
    - Choose the smallest **instance size** (e.g. *512MB*).
    - If given the option, *disable* **Zone Redundancy**.
    - Use an existing **Storage Account** or create a new one.
    - *Configure* **diagnostic settings** *now*, leaving the default.
    - Leave the *defaults* for OpenAI (disable)
    - Leave the *defaults* for Networking (public enabled, virtual disabled).
    - Leave the *defaults* for Monitoring (enable in a supported region).
    - Enable **Continuous Deployment** and point to your repo, signing in to GitHub.
    - Enable **Basic Authentication**.
    - Leave the *defaults* for Authentication (secrets) and tags (none).
    - Wait until the deployment completes.

5. **In the Codespace**, in **Source Control**, click on the *Pull* icon,
or chose it through the `...` menu.
    - Ensure you have the latest changes from GitHub (a new workflow file)
        - If not, click *Pull* again until you do.
    - Edit the **newly** created workflow file in `.github/workflows/`
    i.e. not the existing `quality.yaml`
        - Find the `Install dependencies` step and add `-t .` to the commmand:
        
            ```yaml
            run: pip install -r requirements.txt -t .
            ```
        - Find the `Deploy to Azure Functions` and add `sku: 'flexconsumption' to the with block:
        
            ```yaml
            ...
            with:
              sku: 'flexconsumption'
              app-name: ...
            ```
    - **Commit and Sync** all your changes!

6. **On your repository on GitHub**, go to the *Actions* tab.
    - Wait for the workflow to complete successfully.

7. **In the Azure Portal**, navigate to your Function App.
    - Select your function from the list.
    - Test the function using the *Test/Run* option in the Function
        - Provide the required parameters (tenure, monthly, techsupport)
        - Run the test and check the output for the prediction result.

