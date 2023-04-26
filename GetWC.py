import numpy as np

def GetWC(qC):
    # refer to eq. (9)
    # GetWC calculates Wc.
    # Wc is a 4*16 matrix.

    qC0, qC1, qC2, qC3 = qC
    WC = np.array([[qC0, -qC1, -qC2, -qC3, -qC1, -qC0, qC3, -qC2, -qC2, -qC3, -qC0, qC1, -qC3, qC2, -qC1, -qC0],
                   [qC1, qC0, -qC3, qC2, qC0, -qC1, -qC2, -qC3, qC3, -qC2, qC1, qC0, -qC2, -qC3, -qC0, qC1],
                   [qC2, qC3, qC0, -qC1, -qC3, qC2, -qC1, -qC0, qC0, -qC1, -qC2, -qC3, qC1, qC0, -qC3, qC2],
                   [qC3, -qC2, qC1, qC0, qC2, qC3, qC0, -qC1, -qC1, -qC0, qC3, -qC2, qC0, -qC1, -qC2, -qC3]])
    
    return WC