use ndarray::{Array, Array1, ArrayD, ArrayViewD, ArrayView2, Array2, ArrayViewMutD, ArrayViewMut2};
use numpy::{IntoPyArray, PyArrayDyn, PyArray2, PyReadonlyArrayDyn, PyReadonlyArray2};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn pyconcord(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // immutable example
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // mutable example (no return)
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    fn trace_triple_mult(a: ArrayView2<'_, f64>, 
					    b: ArrayView2<'_, f64>, 
					    c: ArrayView2<'_, f64>, 
					    p: &usize) -> f64 {
		let mut result: f64 = 0.0;
		for i in 0..*p {
			for j in 0..*p {
				for k in 0..*p {
					result += a[[i, j]] * b[[j, k]] * c[[k, i]];
				}
			}
		}
		result / 2.0
	}

	fn sign(a: f64) -> f64 {
		if ( a > 0.0 ) {
			return 1.0
		} else {
			return - 1.0
		}
	}

	
	fn max_zero(a: f64) -> f64 {
		if a > 0.0 {
			return a
		} else {
			return 0.0
		}
	}	

	fn abs(a: f64) -> f64 {
		if a > 0.0 {
			return a
		} else {
			return - a
		}
	}

	fn sthreshmat<'a>(mut omega: &'a Array2<f64>, t: ArrayView2<'_, f64>, tau: f64) {
		let sign_omega = omega.mapv(sign);
		let mut abs_omega = omega.mapv(abs).to_owned();
		abs_omega = abs_omega - tau * &t;
		abs_omega.mapv_inplace(max_zero);
		omega = &(&sign_omega * &abs_omega.view());
	}

    fn concord(data: ArrayView2<'_, f64>, n: &usize, _p: &usize, _lambda: f64) -> Array2<f64> {
        
        let maxitr = 100;
        // Initial guess - identity matrix
        let omega_k = Array2::<f64>::eye(* _p);

        let s = data.t().dot(&data) / (*n as f64);

        let ttm = trace_triple_mult(omega_k.view(), s.view(), omega_k.view(), _p);
        println!("ttm: {:?}", ttm);

        let g_k = s.clone() - omega_k.view(); // simplified since x is the identity
        let mut lambda_matrix = Array2::<f64>::eye(* _p) * _lambda;
        let mut diagonal = lambda_matrix.diag_mut();
        diagonal = Array1::<f64>::zeros(*_p).view_mut();
        let mut not_converged = true;
        let mut tau: f64;
        let mut taun = 1.0;
        let mut diagitr: i16 = 0;
        let mut backitr: i16 = 0;
        let mut itr: i16 = 0;
        let half = 0.5;

        while not_converged {

        	tau = taun.clone();

        	if diagitr != 0 || backitr != 0 {
        		tau = tau * half;
        	}

        	let mut tmp = omega_k.to_owned();
        	tmp = tmp - tau * &g_k;

        	itr = itr + 1;
        	if itr >= maxitr {
        		not_converged = false;
        	}
        	not_converged = false;
        };


        s
    }

    #[pyfn(m, "concord")]
    fn concord_py<'py>(
    	py: Python<'py>,
    	x: PyReadonlyArray2<f64>,
    	lambda: f64
    	) -> &'py PyArray2<f64> {
		let shape = x.shape();
		concord(x.as_array(), &shape[0], &shape[1], lambda).into_pyarray(py)
    }

    // wrapper of `axpy`
    #[pyfn(m, "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        axpy(a, x, y).into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m, "mult")]
    fn mult_py(_py: Python<'_>, a: f64, x: &PyArrayDyn<f64>) -> PyResult<()> {
        let x = unsafe { x.as_array_mut() };
        mult(a, x);
        Ok(())
    }

    Ok(())
}