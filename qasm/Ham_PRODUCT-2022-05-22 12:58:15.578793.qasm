OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
s q[2];
h q[2];
cx q[2],q[9];
s q[3];
h q[3];
cx q[3],q[9];
s q[7];
h q[7];
cx q[7],q[9];
cx q[8],q[9];
rz(0.009605064) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[3],q[9];
cx q[2],q[9];
s q[9];
h q[9];
cx q[9],q[2];
cx q[8],q[2];
s q[7];
h q[7];
cx q[7],q[2];
s q[3];
h q[3];
cx q[3],q[2];
rz(0.009605064) q[2];
cx q[3],q[2];
cx q[7],q[2];
h q[7];
sdg q[7];
cx q[8],q[2];
cx q[9],q[2];
h q[9];
sdg q[9];
rz(2.1418549) q[9];
s q[4];
h q[4];
cx q[4],q[1];
s q[3];
h q[3];
cx q[3],q[1];
rz(0.0072404975) q[1];
cx q[3],q[1];
cx q[4],q[1];
h q[4];
sdg q[4];
s q[1];
h q[1];
cx q[1],q[6];
cx q[2],q[6];
s q[3];
h q[3];
cx q[3],q[6];
s q[5];
h q[5];
cx q[5],q[6];
rz(-0.0069327837) q[6];
cx q[5],q[6];
cx q[3],q[6];
h q[3];
sdg q[3];
cx q[2],q[6];
cx q[1],q[6];
cx q[9],q[4];
rz(0.24714174) q[4];
cx q[9],q[4];
cx q[0],q[3];
rz(0.10785972) q[3];
cx q[0],q[3];
h q[9];
cx q[9],q[0];
h q[8];
cx q[8],q[0];
rz(0.021080375) q[0];
cx q[8],q[0];
h q[8];
cx q[9],q[0];
h q[9];
s q[1];
h q[1];
cx q[1],q[6];
cx q[2],q[6];
s q[3];
h q[3];
cx q[3],q[6];
s q[5];
h q[5];
cx q[5],q[6];
rz(-0.0069327837) q[6];
cx q[5],q[6];
cx q[3],q[6];
h q[3];
sdg q[3];
cx q[2],q[6];
cx q[1],q[6];
h q[1];
sdg q[1];
cx q[9],q[4];
rz(0.24714174) q[4];
cx q[9],q[4];
h q[0];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
h q[3];
cx q[3],q[8];
rz(-0.018024086) q[8];
cx q[3],q[8];
h q[3];
cx q[2],q[8];
cx q[1],q[8];
cx q[0],q[8];
h q[0];
s q[9];
h q[9];
cx q[9],q[5];
cx q[8],q[5];
cx q[7],q[5];
rz(-0.0042997153) q[5];
cx q[7],q[5];
cx q[8],q[5];
cx q[9],q[5];
s q[3];
h q[3];
cx q[3],q[9];
s q[4];
h q[4];
cx q[4],q[9];
s q[5];
h q[5];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(-0.016746723) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
cx q[5],q[9];
cx q[4],q[9];
cx q[3],q[9];
s q[9];
h q[9];
cx q[9],q[3];
cx q[8],q[3];
cx q[7],q[3];
cx q[6],q[3];
s q[5];
h q[5];
cx q[5],q[3];
s q[4];
h q[4];
cx q[4],q[3];
rz(-0.016746723) q[3];
cx q[4],q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[7],q[3];
cx q[8],q[3];
cx q[9],q[3];
rz(2.1418549) q[9];
s q[4];
h q[4];
cx q[4],q[1];
s q[3];
h q[3];
cx q[3],q[1];
rz(0.0072404975) q[1];
cx q[3],q[1];
h q[3];
sdg q[3];
cx q[4],q[1];
h q[4];
sdg q[4];
cx q[3],q[9];
s q[5];
h q[5];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(0.0047358738) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
cx q[5],q[9];
cx q[3],q[9];
s q[9];
h q[9];
cx q[9],q[3];
cx q[8],q[3];
cx q[7],q[3];
cx q[6],q[3];
s q[5];
h q[5];
cx q[5],q[3];
rz(0.0047358738) q[3];
cx q[5],q[3];
h q[5];
sdg q[5];
cx q[6],q[3];
cx q[7],q[3];
cx q[8],q[3];
cx q[9],q[3];
h q[9];
sdg q[9];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
cx q[7],q[3];
rz(0.14110864) q[3];
cx q[7],q[3];
cx q[0],q[3];
rz(0.10785972) q[3];
cx q[0],q[3];
h q[9];
cx q[9],q[0];
h q[8];
cx q[8],q[0];
rz(0.021080375) q[0];
cx q[8],q[0];
cx q[9],q[0];
h q[3];
cx q[3],q[9];
h q[4];
cx q[4],q[9];
h q[8];
cx q[8],q[9];
rz(0.0060693137) q[9];
cx q[8],q[9];
cx q[4],q[9];
h q[4];
cx q[3],q[9];
h q[3];
cx q[6],q[0];
rz(0.13464686) q[0];
cx q[6],q[0];
cx q[0],q[3];
rz(0.10785972) q[3];
cx q[0],q[3];
h q[9];
cx q[9],q[0];
h q[8];
cx q[8],q[0];
rz(0.021080375) q[0];
cx q[8],q[0];
h q[8];
cx q[9],q[0];
s q[2];
h q[2];
cx q[2],q[7];
cx q[3],q[7];
s q[4];
h q[4];
cx q[4],q[7];
h q[5];
cx q[5],q[7];
cx q[6],q[7];
rz(-0.009759481) q[7];
cx q[6],q[7];
cx q[5],q[7];
h q[5];
cx q[4],q[7];
h q[4];
sdg q[4];
cx q[3],q[7];
cx q[2],q[7];
h q[2];
sdg q[2];
cx q[8],q[3];
rz(0.16940784) q[3];
cx q[8],q[3];
cx q[0],q[3];
rz(0.10785972) q[3];
cx q[0],q[3];
h q[9];
cx q[9],q[0];
h q[8];
cx q[8],q[0];
rz(0.021080375) q[0];
cx q[8],q[0];
h q[8];
cx q[9],q[0];
h q[9];
h q[3];
cx q[3],q[5];
h q[4];
cx q[4],q[5];
rz(0.021080375) q[5];
cx q[4],q[5];
h q[4];
cx q[3],q[5];
h q[3];
s q[9];
h q[9];
cx q[9],q[0];
s q[8];
h q[8];
cx q[8],q[0];
rz(0.021080375) q[0];
cx q[8],q[0];
h q[8];
sdg q[8];
cx q[9],q[0];
h q[9];
sdg q[9];
cx q[0],q[3];
rz(0.10785972) q[3];
cx q[0],q[3];
h q[9];
cx q[9],q[0];
h q[8];
cx q[8],q[0];
rz(0.021080375) q[0];
cx q[8],q[0];
cx q[9],q[0];
h q[9];
cx q[3],q[8];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-0.018024086) q[8];
cx q[7],q[8];
cx q[6],q[8];
cx q[5],q[8];
cx q[3],q[8];
h q[8];
cx q[8],q[3];
cx q[7],q[3];
cx q[6],q[3];
h q[5];
cx q[5],q[3];
rz(-0.018024086) q[3];
cx q[5],q[3];
h q[5];
cx q[6],q[3];
cx q[7],q[3];
cx q[8],q[3];
h q[8];
s q[0];
h q[0];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
s q[4];
h q[4];
cx q[4],q[5];
rz(-0.070233541) q[5];
cx q[4],q[5];
cx q[3],q[5];
cx q[2],q[5];
cx q[1],q[5];
cx q[0],q[5];
s q[9];
h q[9];
cx q[9],q[6];
s q[8];
h q[8];
cx q[8],q[6];
rz(0.0072404975) q[6];
cx q[8],q[6];
h q[8];
sdg q[8];
cx q[9],q[6];
s q[2];
h q[2];
cx q[2],q[7];
cx q[3],q[7];
s q[4];
h q[4];
cx q[4],q[7];
h q[5];
cx q[5],q[7];
cx q[6],q[7];
rz(-0.009759481) q[7];
cx q[6],q[7];
cx q[5],q[7];
h q[5];
cx q[4],q[7];
cx q[3],q[7];
cx q[2],q[7];
h q[2];
sdg q[2];
cx q[8],q[3];
rz(0.16940784) q[3];
cx q[8],q[3];
s q[0];
h q[0];
cx q[0],q[5];
cx q[1],q[5];
cx q[2],q[5];
cx q[3],q[5];
s q[4];
h q[4];
cx q[4],q[5];
rz(-0.070233541) q[5];
cx q[4],q[5];
h q[4];
sdg q[4];
cx q[3],q[5];
cx q[2],q[5];
cx q[1],q[5];
cx q[0],q[5];
h q[0];
sdg q[0];
s q[9];
h q[9];
cx q[9],q[6];
s q[8];
h q[8];
cx q[8],q[6];
rz(0.0072404975) q[6];
cx q[8],q[6];
cx q[9],q[6];
h q[9];
sdg q[9];
h q[0];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
h q[4];
cx q[4],q[8];
s q[5];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-0.015528882) q[8];
cx q[7],q[8];
cx q[6],q[8];
cx q[5],q[8];
h q[5];
sdg q[5];
cx q[4],q[8];
h q[4];
cx q[3],q[8];
cx q[2],q[8];
cx q[1],q[8];
cx q[0],q[8];
cx q[9],q[2];
rz(0.13757105) q[2];
cx q[9],q[2];
s q[2];
h q[2];
cx q[2],q[9];
cx q[3],q[9];
s q[4];
h q[4];
cx q[4],q[9];
h q[7];
cx q[7],q[9];
cx q[8],q[9];
rz(0.011993522) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[4],q[9];
cx q[3],q[9];
cx q[2],q[9];
h q[9];
cx q[9],q[5];
cx q[8],q[5];
cx q[6],q[5];
rz(-0.0042997153) q[5];
cx q[6],q[5];
cx q[8],q[5];
cx q[9],q[5];
h q[9];
s q[2];
h q[2];
cx q[2],q[7];
cx q[3],q[7];
s q[4];
h q[4];
cx q[4],q[7];
h q[5];
cx q[5],q[7];
cx q[6],q[7];
rz(-0.009759481) q[7];
cx q[6],q[7];
cx q[5],q[7];
cx q[4],q[7];
h q[4];
sdg q[4];
cx q[3],q[7];
cx q[2],q[7];
h q[2];
sdg q[2];
cx q[8],q[3];
rz(0.16940784) q[3];
cx q[8],q[3];
h q[0];
cx q[0],q[8];
cx q[1],q[8];
cx q[2],q[8];
cx q[3],q[8];
h q[4];
cx q[4],q[8];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-0.015528882) q[8];
cx q[7],q[8];
cx q[6],q[8];
cx q[5],q[8];
cx q[4],q[8];
cx q[3],q[8];
cx q[2],q[8];
cx q[1],q[8];
cx q[0],q[8];
h q[8];
cx q[8],q[0];
cx q[7],q[0];
cx q[6],q[0];
h q[5];
cx q[5],q[0];
h q[4];
cx q[4],q[0];
cx q[3],q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(-0.015528882) q[0];
cx q[1],q[0];
cx q[2],q[0];
cx q[3],q[0];
cx q[4],q[0];
cx q[5],q[0];
cx q[6],q[0];
cx q[7],q[0];
cx q[8],q[0];
cx q[3],q[8];
h q[5];
cx q[5],q[8];
cx q[6],q[8];
cx q[7],q[8];
rz(-0.018024086) q[8];
cx q[7],q[8];
cx q[6],q[8];
cx q[5],q[8];
cx q[3],q[8];
h q[8];
cx q[8],q[3];
cx q[7],q[3];
cx q[6],q[3];
h q[5];
cx q[5],q[3];
rz(-0.018024086) q[3];
cx q[5],q[3];
cx q[6],q[3];
cx q[7],q[3];
cx q[8],q[3];
h q[8];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
cx q[7],q[3];
rz(0.14110864) q[3];
cx q[7],q[3];
h q[0];
cx q[0],q[4];
cx q[1],q[4];
cx q[3],q[4];
rz(-0.0042997153) q[4];
cx q[3],q[4];
cx q[1],q[4];
cx q[0],q[4];
h q[0];
cx q[9],q[3];
rz(0.11331214) q[3];
cx q[9],q[3];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
cx q[7],q[3];
rz(0.14110864) q[3];
cx q[7],q[3];
cx q[0],q[9];
rz(0.22867909) q[9];
cx q[0],q[9];
cx q[7],q[1];
rz(0.13960362) q[1];
cx q[7],q[1];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
cx q[7],q[3];
rz(0.14110864) q[3];
cx q[7],q[3];
cx q[0],q[9];
h q[5];
cx q[5],q[9];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(-0.070233541) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
cx q[5],q[9];
cx q[0],q[9];
h q[9];
cx q[9],q[0];
cx q[8],q[0];
cx q[7],q[0];
cx q[6],q[0];
h q[5];
cx q[5],q[0];
rz(-0.070233541) q[0];
cx q[5],q[0];
h q[5];
cx q[6],q[0];
cx q[7],q[0];
cx q[8],q[0];
cx q[9],q[0];
cx q[3],q[7];
rz(0.14110864) q[7];
cx q[3],q[7];
cx q[7],q[3];
rz(0.14110864) q[3];
cx q[7],q[3];
s q[0];
h q[0];
cx q[0],q[9];
s q[1];
h q[1];
cx q[1],q[9];
h q[6];
cx q[6],q[9];
cx q[7],q[9];
cx q[8],q[9];
rz(-0.009759481) q[9];
cx q[8],q[9];
cx q[7],q[9];
cx q[6],q[9];
cx q[1],q[9];
cx q[0],q[9];
h q[9];
cx q[9],q[0];
cx q[8],q[0];
cx q[7],q[0];
h q[6];
cx q[6],q[0];
s q[1];
h q[1];
cx q[1],q[0];
rz(-0.009759481) q[0];
cx q[1],q[0];
h q[1];
sdg q[1];
cx q[6],q[0];
h q[6];
cx q[7],q[0];
cx q[8],q[0];
cx q[9],q[0];
h q[9];
rz(2.1418549) q[9];
s q[4];
h q[4];
cx q[4],q[1];
s q[3];
h q[3];
cx q[3],q[1];
rz(0.0072404975) q[1];
cx q[3],q[1];
h q[3];
sdg q[3];
cx q[4],q[1];
h q[4];
sdg q[4];
rz(-0.84896351) q[1];
s q[8];
h q[8];
cx q[8],q[2];
s q[7];
h q[7];
cx q[7],q[2];
h q[4];
cx q[4],q[2];
cx q[3],q[2];
rz(0.009605064) q[2];
cx q[3],q[2];
cx q[4],q[2];
h q[4];
cx q[7],q[2];
cx q[8],q[2];
rz(2.1418549) q[9];
s q[4];
h q[4];
cx q[4],q[1];
s q[3];
h q[3];
cx q[3],q[1];
rz(0.0072404975) q[1];
cx q[3],q[1];
cx q[4],q[1];
s q[0];
h q[0];
cx q[0],q[8];
cx q[1],q[8];
s q[2];
h q[2];
cx q[2],q[8];
s q[7];
h q[7];
cx q[7],q[8];
rz(-0.0069327837) q[8];
cx q[7],q[8];
cx q[2],q[8];
cx q[1],q[8];
cx q[0],q[8];
s q[8];
h q[8];
cx q[8],q[0];
s q[7];
h q[7];
cx q[7],q[0];
s q[2];
h q[2];
cx q[2],q[0];
cx q[1],q[0];
rz(-0.0069327837) q[0];
cx q[1],q[0];
cx q[2],q[0];
h q[2];
sdg q[2];
cx q[7],q[0];
h q[7];
sdg q[7];
cx q[8],q[0];
h q[8];
sdg q[8];
rz(2.1418549) q[9];
s q[4];
h q[4];
cx q[4],q[1];
s q[3];
h q[3];
cx q[3],q[1];
rz(0.0072404975) q[1];
cx q[3],q[1];
cx q[4],q[1];
h q[4];
sdg q[4];
s q[0];
h q[0];
cx q[0],q[9];
cx q[1],q[9];
cx q[2],q[9];
s q[3];
h q[3];
cx q[3],q[9];
rz(0.025467828) q[9];
cx q[3],q[9];
cx q[2],q[9];
cx q[1],q[9];
cx q[0],q[9];
cx q[9],q[0];
s q[3];
h q[3];
cx q[3],q[0];
cx q[2],q[0];
cx q[1],q[0];
rz(0.025467828) q[0];
cx q[1],q[0];
cx q[2],q[0];
cx q[3],q[0];
h q[3];
sdg q[3];
cx q[9],q[0];
h q[3];
cx q[3],q[9];
h q[4];
cx q[4],q[9];
h q[8];
cx q[8],q[9];
rz(0.0060693137) q[9];
cx q[8],q[9];
h q[8];
cx q[4],q[9];
cx q[3],q[9];
h q[3];
cx q[6],q[0];
rz(0.13464686) q[0];
cx q[6],q[0];
cx q[0],q[9];
rz(0.22867909) q[9];
cx q[0],q[9];
cx q[7],q[1];
rz(0.13960362) q[1];
cx q[7],q[1];
rz(-0.84896351) q[1];
s q[8];
h q[8];
cx q[8],q[2];
s q[7];
h q[7];
cx q[7],q[2];
h q[4];
cx q[4],q[2];
cx q[3],q[2];
rz(0.009605064) q[2];
cx q[3],q[2];
cx q[4],q[2];
cx q[7],q[2];
h q[7];
sdg q[7];
cx q[8],q[2];
h q[8];
sdg q[8];
h q[0];
cx q[0],q[4];
cx q[1],q[4];
cx q[3],q[4];
rz(-0.0042997153) q[4];
cx q[3],q[4];
cx q[1],q[4];
cx q[0],q[4];
h q[0];
cx q[9],q[3];
rz(0.11331214) q[3];
cx q[9],q[3];
