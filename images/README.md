# How to read the test images

The first folder name is for the model; for example, multi-fidelity-gp refers to the model called multi-fidelity-gp.

The second folder name is the base kernel. This is the kernel that acts on the inputs. The kernel on the outputs is always the Coregion kernel.

The third folder is the likelihood used to get the variances.

The file name is the number of high fidelity samples and the number of low fidelity samples, separated by an underscore.

The plots themselves show the predicted function (labeled Y1, Y2, and Y3) as well as the actual function, evaluated at 1000 test points. Ideally, the predicted function will perfectly match the real function, but at the least the variance should cover every point.