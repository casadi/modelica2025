import time
import shutil
import os

def compile_modelica(model, mofile, compiler):
  """
  Compile a simple Modelica model contained in a single file
  """
  mofile = os.path.abspath(mofile)
  modirectory = os.path.dirname(mofile)
  mofile = os.path.basename(mofile)
  try:
    backup_dir = os.getcwd()
    os.chdir(modirectory)
    # Start timer
    t1 = time.time()
    # Compile
    if compiler == 'OpenModelica':
        # Compile using OpenModelica, cf. the OpenModelica User's Guide
        from OMPython import OMCSessionZMQ
        # Load the Modelica model
        omc = OMCSessionZMQ()
        if omc.loadFile(mofile).startswith('false'):
          raise Exception('Modelica compilation failed: {}'.format(omc.sendExpression('getErrorString()')))
        # Enable analytic derivatives
        omc.sendExpression('setDebugFlags("-disableDirectionalDerivatives")')
        # Generate the FMU
        fmu_file = omc.sendExpression(f'translateModelFMU({model})')
        flag = omc.sendExpression('getErrorString()')
        if not fmu_file.endswith('.fmu'): raise Exception(f'FMU generation failed: {flag}')
        print(f"translateModelFMU warnings:\n{flag}")
    elif compiler == 'OCT':
        # Compile using OCT, cf. OPTIMICA Compiler Toolkit (OCT) User's Guide
        from pymodelica import compile_fmu
        compiler_options = dict(generate_ode_jacobian = True)
        fmu_file = compile_fmu(model, [mofile], compiler_options = compiler_options)
    else:
        raise Exception(f'Unknown compiler: {compiler}')
    # Output results of compilation
    print(f'Compiled {mofile} into {fmu_file} using {compiler} in {time.time() - t1} s')
    return fmu_file
  finally:
    os.chdir(backup_dir)

def unpack_fmu(fmu_file):
  """
  Unpack the contents of an FMU file
  """
  # To create a directory, strip the .fmu ending from the fmu_file and add a timestamp
  from datetime import datetime
  suffix = datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
  unpacked_fmu = os.path.join(os.getcwd(),fmu_file[:fmu_file.find('.')] + suffix)
  # Unzip
  import zipfile
  with zipfile.ZipFile(fmu_file, 'r') as zip_ref: zip_ref.extractall(unpacked_fmu)
  print(f'Unpacked {fmu_file} into {unpacked_fmu}')
  return unpacked_fmu

def cleanup_fmu(unpacked_fmu):
  shutil.rmtree(unpacked_fmu)