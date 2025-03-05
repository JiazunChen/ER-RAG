## Adding your runtime

This repository is a valid submission (and submission structure). 
You can simply add your dependencies on top of this repository.

Few of the most common ways are as follows:

* `requirements.txt` -- The `pip3` packages used by your inference code. As you add new pip3 packages to your inference procedure either manually add them to `requirements.txt` or if your software runtime is simple, perform:
    ```
    # Put ALL of the current pip3 packages on your system in the submission
    >> pip3 freeze >> requirements.txt
    >> cat requirements.txt
    aicrowd_api
    coloredlogs
    matplotlib
    pandas
    [...]
    ```

* `apt.txt` -- The Debian packages (via aptitude) used by your inference code!

These files are used to construct your **AIcrowd submission docker containers** in which your code will run.

* `Dockerfile` -- **For advanced users only**. `Dockerfile` gives you more flexibility on defining the software runtime used during evaluations.

----

To test your image builds locally, you can use the included [docker_run.sh](../docker_run.sh) script.
