use pyo3::{prelude::*, types::PyDict};
use pyo3_tch::PyTensor;
use tch::{nn::*, *};

#[pyfunction]
#[pyo3(name = "forward")]
fn exp_forward(x: PyTensor) -> PyTensor {
    // TODO call metal kernel
    PyTensor(x.0.exp())
}

#[pyfunction]
#[pyo3(name = "backward")]
fn exp_backward(x: PyTensor, grad: PyTensor) -> PyTensor {
    // TODO call metal kernel
    PyTensor(x.0 * grad.0)
}

fn exp_module() -> impl Fn(Tensor) -> Tensor {
    init_tensor_op1("exp", |m| {
        m.add_function(wrap_pyfunction!(exp_forward, m)?)?;
        m.add_function(wrap_pyfunction!(exp_backward, m)?)?;
        Ok(())
    })
}

/// Create a function that modifies 1 tensor and returns 1 tensor.
/// The forward() and backward() functions are registered in the fn_register closure.
fn init_tensor_op1(
    name: impl AsRef<str>,
    fn_register: impl Fn(&PyModule) -> PyResult<()>,
) -> impl Fn(Tensor) -> Tensor {
    let fun = init_pymodule(name, fn_register).expect("failed to initialize module");

    move |tensor: Tensor| {
        Python::with_gil(|py| {
            fun.call1(py, (PyTensor(tensor),))
                .expect("failed to apply")
                .extract::<PyTensor>(py)
                .expect("it wasn't a PyTensor??")
                .0
        })
    }
}

const KERNEL_AUTOGRAD_PY_TEMPLATE: &str = include_str!("kernel_autograd.py");

/// Initialize a Python module with a given name and a function to register pyfunctions in it.
fn init_pymodule(
    name: impl AsRef<str>,
    fn_register: impl Fn(&PyModule) -> PyResult<()>,
) -> PyResult<PyObject> {
    let name = name.as_ref();
    Python::with_gil(|py| {
        // create a module and register the functions
        let module = PyModule::new(py, name)?;
        fn_register(&module)?;

        // insert into sys.modules
        let sys = PyModule::import(py, "sys")?;
        let py_modules: &PyDict = sys.getattr("modules")?.downcast()?;
        py_modules.set_item(name, module)?;

        let function_wrapper_source = KERNEL_AUTOGRAD_PY_TEMPLATE.replace("NEW", name);

        // create another module that contains the generated toch.autograd.Function in it
        let function_module = PyModule::from_code(
            py,
            &function_wrapper_source,
            &format!("{}_function_wrapper.py", name),
            &format!("{}_function_wrapper", name),
        )?;

        let fun: Py<PyAny> = function_module
            .getattr(format!("{}_wrapper", name).as_str())?
            .into();

        Ok(fun)
    })
}

#[test]
fn test_exp_module() {
    pyo3::prepare_freethreaded_python();
    let _ = exp_module();
}
