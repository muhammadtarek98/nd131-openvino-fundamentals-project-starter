#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.core=IEcore()
        self.network=None
        self.exec_network = None
        self.input_frame=None
        self.output=None
        self.inference_request=None

    def load_model(self,model,device,cpu_ext=None,num_requests=0):
        ### TODO: Load the model ###
        modelxml=model
        weights=os.path.splitext(modelxml)[0] + ".bin"
        self.core = IECore()
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)  
        self.network = IENetwork(model=modelxml, weights=weights)
        ### TODO: Check for supported layers ###
        if "CPU" in device:
            supported_layers = self.plugin.query_network(self.network, "CPU")
            not_supported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")
                sys.exit(1)
        ### TODO: Add any necessary extensions ###
        self.exec_network = self.plugin.load_network(self.network, device,num_requests=num_requests)
        ### TODO: Return the loaded inference plugin ###
        self.input_frame = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))
        ### Note: You may need to update the function parameters. ###
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
         return self.network.inputs[self.input_frame].shape
      
      def get_output_name(self):
        output, _ = "", self.network.outputs[next(iter(self.network.outputs.keys()))]
        for output_key in self.network.outputs:
            if self.network.layers[output_key].type == "DetectionOutput":
                output, _ = output_key, self.network.outputs[output_key]
        
        if output == "":
            log.error("Can't find a DetectionOutput layer in the topology")
            exit(-1)
        return output

    def exec_net(self,image, request_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id=request_id, 
            inputs={self.input_frame: image})
        return

    def wait(self,request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
         return self.exec_network.requests[0].outputs[self.output]
        ### Note: You may need to update the function parameters. ###
