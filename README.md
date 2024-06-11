This is for the APML project:

Please make sure to import the necessary data. These are just the ML scripts with imports that fit my personal data in the pd.read_csv("") arguments.

The "..._novel" files include a version of the training that generates great results, but not via the original DANN training methodology. Essentially, the backpropagation method is encoded into a single step, and thus, cannot completely seperate the two different backpropagation steps. However, this somehow manages to enhance the performance of the domain classifier, whilst yielding occasionally improved results to the normal DANN.


