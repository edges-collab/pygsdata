# Usage

The primary data object that all functionality is built around is the
{class}`gsdata.GSData` object. This object stores all relevant data from
a global-signal measurement, and has useful methods for accessing parts of the data,
as well as reading/writing the data to standard formats (including HDF5 and ACQ).

The `GSData` object is considered to be immutable. This means that functions that
process the object return a *new* object with updated data. This makes reasoning about
the code much simpler, and makes it easier to write code that is reusable.
All processing functions that take in a GSData object and return a new one are
decorated with the `@gsregister` decorator, which registers the function to be
discoverable by the CLI (more on that later), but also enables automatic updating
of the `history` of the object, so no manual updating of the history is required.

## Reading/Writing Data

To read in data as a GSData object, simply use the `from_file` method:

```python
from gsdata import GSData
data = GSData.from_file('data.acq', telescope_name="EDGES-low")
```

Notice that we passed the telescope name, which is a piece of (optional) metadata that
the ACQ file doesn't store. Any parameter to GSData that ACQ doesn't natively contain
can be passed in this way when constructing the object.
While ACQ is readable, the GSData object supports a native HDF5-based format which is
both faster to read, and is able to contain more metadata. We can write such a file:

```
data.write_gsh5("data.gsh5")
```

This file can be read using the same method as above:

```
data = GSData.from_file('data.gsh5')
```

Notice that here we didn't have to specify the `telescope_name` parameter, because
the file format contains this information.

## Updating the Object

As already stated, the GSData object is to be considered immutable. This means that you
can be confident that any function that "changes" your data object will in fact return
a *new* object with the updated data. Thus, you can keep a reference to the original
unchanged object if necessary. Despite this, if any arrays are the *same* between the
objects, then the memory will not be copied. Thus, if you were to inadvertantly in-place
modify one of the arrays, both objects would be affected. Don't do this.

The "official" way to update the object is to use the `update` method:

```python
data = data.update(data=data.data * 3, data_unit="uncalibrated")
```

This will return the new object. However, this doesn't update the history automatically.
The history can be updated by supplying a dictionary with at least a message:

```python
data = data.update(
    data=data.data * 3,
    data_unit="uncalibrated",
    history={"message": "Multiplied by 3"}
)
```

In actual fact, the history object that is added is a {class}`edges_analysis.gsdata.Stamp`
object, which is a lightweight object that can be easily serialized to YAML, and adds
a default timestamp and set of code versions to the history. You can use one of these
directly if you wish:

```python
from gsdata import Stamp
data = data.update(
    data=data.data * 3,
    data_unit="uncalibrated",
    history=Stamp(message="Multiplied by 3", timestamp=datetime.now())
)
```

If you write a function that updates a GSData object, it is better to include the
function name and the parameters it uses in the history:

```python
def multiply_by_3(data, data_unit):
    return data.update(
        data=data.data * 3,
        data_unit=data_unit,
        history={"function": "multiply_by_3", "parameters": {"data_unit": data_unit}}
    )

data = multiply_by_3(data, "uncalibrated")
```

However, if you are going to write functions that update the data, there is a better way
to do it, as we shall see now.

## Using the Register

There is a decorator defined that makes writing new functions that update GSData objects
simpler, called {func}`gsdata.gsregister`.
This decorator does a few things: it registers the function into a global dictionary,
`GSDATA_PROCESSORS`, and it adds the function to the `history` of the object.
*Using* registered functions is simple: just call the function with the object as the
first argument, and any other parameters as keyword arguments. Since most
internally-defined functions have already been registered, you can use them out of the
box. For example:

```
from gsdata.select import select_freqs
from astropy import units as un

data = select_freqs(data, freq_range=(50*un.MHz, 100*un.MHz))
```

The returned `data` object has a different data-shape (it has frewer frequencies), and
the history contains a new entry. You can print that history:

```
print(str(data.history))
```

Or just print the most recent addition to the history:

```
print(str(data.history[-1]))
```

The `history` attribute also has a {meth}`pretty` method, which can be used with the
rich library to pretty-print the history:

```
from rich.console import Console
console = Console()
console.print(data.history.pretty())
```

Adding your own registered processor is simple -- just use the decorator over a function
with the correct signature:

```python
from gsdata import gsregister, GSData

@gsregister("calibrate")
def pow_data(data: GSData, *, n: int=2) -> GSData:
    return data.update(data=data.data**n)
```

Note here that the first argument to the function is always a GSData instance, and the
return value is always another GSData instance. All other parameters should be keyword
arguments, and can in principle be anything, but it is best to make them types that can
easily be understood by YAML (this helps with writing out the history, and also for
defining workflows for the CLI).
Note also that the `gsregister` decorator takes a single argument: the *kind* of
processor. This is important, because it enables the workflow to make judgments on how
to call the function in certain cases, and also makes it possible to find subsets of the
available processors.


## Making Plots

The {mod}`gsdata.plots` module contains functions that can be used to make plots
from a GSData object. For example, let's say we have a GSData file:

```python
from gsdata import GSData, plots
data = GSData.from_file('2015_202_00.gsh5')

# Plot a flagged waterfall of the data (whether it's residuals or spectra)
plots.plot_waterfall(data)

# Plot the same but show the nsamples intsead of data
plots.plot_waterfall(data, attribute='nsamples')

# Plot the data residuals (if they exist) and don't apply any flags.
plots.plot_waterfall(data, attribute='resids', which_flags=())
```
