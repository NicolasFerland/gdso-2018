# UCB GDSO Workshop: 2018
*A single spot to share code and ideas for the 2018 UCB GDSO workshop.*

The DeepGlobe 2018 Challenge is over and closed, but we can still jump in and play around with it! First, I think we should have a read through [this white paper announcing the challenge](https://www.groundai.com/project/deepglobe-2018-a-challenge-to-parse-the-earth-through-satellite-images/) - it's very helpful for an overview and background on these problems.

Then, we can get training data for the Roads Extraction challenge or the Building Detection challenge from the SpaceNet collection of public data (assembling the data they used for the Land Cover Classification challenge seems a little more difficult, unfortunately).

### Downloading the SpaceNet data:

This is a little involved, since the data is shared through an Amazon Web Services (AWS) account, set up for the downloader to pay any transfer costs that may be incurred.  You can follow these steps though, and send out any questions you may have:

- Create an AWS account with an Access Key ID, if you don't already have one   
  - Go to http://console.aws.amazon.com and create an account if you don't have one
  - Log in to your account, click on "Services" in the upper left, search for the "IAM" service, and go there
  - On the left panel, choose "Users"
  - Add a new user for yourself with "Programmatic Access"
  - On the permissions page, choose "Attach existing policies directly" and choose "AdministratorAccess" (the first one)
  - Download the credentials `.csv` and stash that somewhere for safekeeping.
  - **Important:** be careful with those credentials and keep them private; if anyone finds them, they can use them to charge a lot of AWS fees to your account!
- Install the `aws` command line interface (CLI) if you don't already have it
  - [Install instructions here on the right](https://aws.amazon.com/cli/)
  - Run `aws configure` and enter your `AWS Access Key ID` and `AWS Secret Access Key`, from the IAM user credentials you got above
  - Other entries (default region, et cetera) do not matter; you can leave those blank

The complete SpaceNet data set is truly huge, but it's broken down into 5 different areas of interest.  Let's just focus on one of those, to keep this a little more manageable.  [Let's focus on the Las Vegas AOI](https://spacenetchallenge.github.io/AOI_Lists/AOI_2_Vegas.html).

- Download a small set of sample files (~200 MB):
  - `aws s3 cp s3://spacenet-dataset/spacenet_sample.tar.gz .`
  - `tar -xvf spacenet_sample.tar.gz`
- Drag-and-drop one of the `RGB-PanSharpen` images and the associated `geojson` building footprint files into QGIS to have a look (see below for info on QGIS)
- [QGIS screenshot](https://github.com/ishivvers/gdso-2018/blob/master/data/qgis_vegas_img225.png)

You can follow the commands on the [Las Vegas AOI](https://spacenetchallenge.github.io/AOI_Lists/AOI_2_Vegas.html) page to download full data sets for the two different challenges.

**Note:**
With a new account, you get 15GB of free downloads per month from AWS.  If you download more than that, your credit card will get charged, but it will be only a couple dollars at most: https://aws.amazon.com/s3/pricing/

### Suggestions for using this repository:

- Make a folder for your experimental code; this is where you should keep work that probably only you will use.
- Put any work that others may want to use into the `shared/` folder.
- **Important:** Do not add any large files to this repo!  Please add `readme` files or comments to your code to describe how to download or transform the real data as required, but do not commit any image files with `git`. (*Note:* you can stash images in the `data/` directory if you'd like, where they will be ignored by `git`.)


- **Shared code:** Any time you edit any of the shared code, we can follow these steps (or something similar) to allow multiple people to work on the same code at the same time:
  - Pull any changes others may have made (`git pull`)
  - Create a new branch off of the `master` and make your changes there (`git checkout -b my-new-branch`)
  - Add and commit your changes (`git add ...`, `git commit ...`, et cetera)
  - Check out the `master` branch, pull any changes others may have made in the mean time, and merge your branch into `master` (`git checkout master`, `git pull`, `git merge my-new-branch`)
  - Fix any merge conflicts that may be present ([tutorial on merge conflicts](https://www.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/merge-conflicts))
  - Push the master branch up to github: `git push`
  - If we start running into collisions / problems with this scheme, we can use Pull Requests to help manage those, but I think we probably won't really need to
  - [Helpful guide to the `git` concepts of branches, merges, et cetera](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging)



### Helpful tutorials and links:

- [Introduction to git (there are many good tutorials out there, this is just one of them)](http://gitimmersion.com)

- [Introduction to Keras (there are many good tutorials out there, this is just one of them)](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)


- Python coding standards to follow for shared code:
  - [PEP8](https://www.python.org/dev/peps/pep-0008/)
  - [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)


- A great (free) resource on Neural Networks:
  - https://www.deeplearningbook.org


- A great (not free) resource on Neural Networks, with Keras examples:
  - https://www.manning.com/books/deep-learning-with-python


- In all honesty, running a GPU-enabled machine in the cloud can be a bit of a pain.  Here are a few walk-throughs to get you started, though:
  - [Running in AWS](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html)
  - [Running in paperspace, a slightly cheaper alternative than AWS](https://www.google.com/amp/s/blog.paperspace.com/jupyter-notebook-with-a-gpu-the-easy-way/amp/)



### Software suggestions:
- `QGIS` is a solid open-source georeferenced image viewer and editor: I'd recommend getting that for viewing the data
  - https://qgis.org/


- If you haven't tried Jupyter lab yet (the next generation of Jupyter notebooks), I recommend it.  Really nice when working on remote machines: you can have a single `ssh` connection and then use Jupyter for a terminal, a text editor, and a Python notebook all at once.
  - http://jupyterlab.readthedocs.io/en/stable/


- It's really helpful to have good text editors when working with software. I recommend `Atom`:
  - https://atom.io/
