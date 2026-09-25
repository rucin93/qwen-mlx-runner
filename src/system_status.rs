//! Read-only conditions sampled outside benchmark timers, never per token.
use objc::runtime::{BOOL, Class, Object, YES};
use objc::{msg_send, sel, sel_impl};
use serde_json::{Value, json};

pub fn snapshot() -> Value {
    objc::rc::autoreleasepool(|| {
        let Some(class) = Class::get("NSProcessInfo") else {
            return json!({"thermal_state":null,"low_power_mode":null});
        };
        // Foundation's singleton is alive throughout this autorelease pool.
        // Availability guards avoid sending unsupported selectors on older OSes.
        unsafe {
            let info: *mut Object = msg_send![class, processInfo];
            if info.is_null() {
                return json!({"thermal_state":null,"low_power_mode":null});
            }
            let has_thermal: BOOL = msg_send![info, respondsToSelector:sel!(thermalState)];
            let thermal_code: Option<isize> = if has_thermal == YES {
                // NSProcessInfoThermalState has the NSInteger ABI.
                Some(msg_send![info, thermalState])
            } else {
                None
            };
            let has_power: BOOL = msg_send![info, respondsToSelector:sel!(isLowPowerModeEnabled)];
            let low_power: Option<bool> = if has_power == YES {
                let enabled: BOOL = msg_send![info, isLowPowerModeEnabled];
                Some(enabled == YES)
            } else {
                None
            };
            json!({
                "thermal_state":thermal_code.map(|state| match state {
                    0=>"nominal", 1=>"fair", 2=>"serious", 3=>"critical", _=>"unknown"
                }),
                "thermal_state_code":thermal_code,
                "low_power_mode":low_power,
            })
        }
    })
}
