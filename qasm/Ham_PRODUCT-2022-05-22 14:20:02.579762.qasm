OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
s q[0];
h q[0];
cx q[0],q[4];
cx q[1],q[4];
cx q[2],q[4];
s q[3];
h q[3];
cx q[3],q[4];
rz(0.0087211051) q[4];
s q[3];
h q[3];
cx q[3],q[4];
cx q[2],q[4];
cx q[1],q[4];
s q[0];
h q[0];
cx q[0],q[4];
rz(0.0087211051) q[4];
cx q[3],q[4];
h q[3];
sdg q[3];
cx q[6],q[5];
h q[8];
cx q[8],q[5];
rz(-0.0076365624) q[5];
h q[8];
cx q[8],q[5];
cx q[6],q[5];
rz(-0.0076365624) q[5];
cx q[6],q[5];
cx q[8],q[5];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
cx q[6],q[5];
h q[8];
cx q[8],q[5];
rz(-0.0076365624) q[5];
h q[8];
cx q[8],q[5];
cx q[6],q[5];
rz(-0.0076365624) q[5];
s q[3];
h q[3];
cx q[3],q[5];
s q[4];
h q[4];
cx q[4],q[5];
rz(0.021080375) q[5];
cx q[4],q[5];
h q[4];
sdg q[4];
cx q[3],q[5];
s q[3];
h q[3];
cx q[3],q[4];
cx q[2],q[4];
cx q[1],q[4];
s q[0];
h q[0];
cx q[0],q[4];
rz(0.0087211051) q[4];
cx q[0],q[4];
h q[0];
sdg q[0];
cx q[1],q[4];
cx q[2],q[4];
cx q[3],q[4];
h q[3];
sdg q[3];
cx q[7],q[3];
rz(0.14110864) q[3];
cx q[7],q[3];
h q[8];
cx q[8],q[5];
cx q[6],q[5];
rz(-0.0076365624) q[5];
cx q[6],q[5];
cx q[8],q[5];
h q[8];
