// Validate .glb files with the Khronos glTF Validator and print one JSON report per file.
//
//   node validate.mjs a.glb b.glb   ->  {"a.glb": {"errors": 0, "warnings": 0, "infos": 0,
//                                        "messages": [...]}, ...}
//
// The exit status is 1 if any file has an error. Messages carry every issue of severity error or
// warning (severity 0 or 1), with its code, JSON pointer and text.
import { readFileSync } from "node:fs";
import validator from "gltf-validator";

const reports = {};
let errors = 0;
for (const path of process.argv.slice(2)) {
  const report = await validator.validateBytes(new Uint8Array(readFileSync(path)), {
    maxIssues: 100,
  });
  const issues = report.issues;
  reports[path] = {
    errors: issues.numErrors,
    warnings: issues.numWarnings,
    infos: issues.numInfos,
    messages: issues.messages
      .filter((m) => m.severity <= 1)
      .map((m) => ({ severity: m.severity, code: m.code, pointer: m.pointer, message: m.message })),
  };
  errors += issues.numErrors;
}
console.log(JSON.stringify(reports, null, 2));
process.exit(errors ? 1 : 0);
