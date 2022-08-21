# Better Solar API Testing
Quick repo to test out the Better Solar model API, newly separated from our application! This is to get an idea of how a customer may integrate our trained models into their pre-existing workflows.

## Getting Started
* Please add the given API_IP address from the user software instructions sent to line 12 of _api.py_.
* Set up python environment with required packages, see instructions below.
* Hit 'Run' on _main.py_ (or type ``` python main.py ```) in valid Python environment.

_* Note: No need to modify the username in main.py for testing._

## Installation

To run this project, install it locally on your machine. Use:
* Using GitHub CLI:
```sh 
gh repo clone BetterSolarLLC/API_Testing
```
* Using git in terminal/command line:
```sh
git clone https://github.com/BetterSolarLLC/API_Testing.git
```
* Using python IDE w/ integrated source control/git support:
  * Find git clone support, paste ``` https://github.com/BetterSolarLLC/API_Testing.git ```

Using the included `requirements.txt` file, use terminal to easily install dependencies onto your dedicated environment with:
```sh
pip install -r requirements.txt
```

## How it Works
_inference\_pipeline()_: main function, sends the image and required metadata to our processing API.

_display\_output()_: only used to parse the outputs into a user-friendly display. This will most likely be customized by a user to fit their existing software.

The _main.py_ script is set up to run the included example module and cell. Feel free to try this out with the other modules sent for testing. The only changes you should need to make are in _main.py_.

## Contact Us
Email us at _contact@bettersolargroup.com_. 
