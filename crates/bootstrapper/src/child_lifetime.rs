//! Ties child processes to bootstrapper lifetime (job object on Windows, `PR_SET_PDEATHSIG` on Linux).
//!
//! On **macOS** and other Unix systems besides Linux, no extra mechanism is applied via
//! [`ChildLifetimeGroup::prepare_command`] today; children still exit when explicitly killed or
//! when the bootstrapper exits if the platform inherits default process-group behavior.

use std::io;
use std::process::{Child, Command};

#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
#[cfg(windows)]
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_BASIC_LIMIT_INFORMATION,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// Holds OS resources so direct children are terminated when the bootstrapper exits unexpectedly.
pub struct ChildLifetimeGroup {
    #[cfg(windows)]
    job: JobHandle,
}

#[cfg(windows)]
struct JobHandle(HANDLE);

#[cfg(windows)]
impl Drop for JobHandle {
    fn drop(&mut self) {
        if self.0 != 0 && self.0 != -1_isize as HANDLE {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

impl ChildLifetimeGroup {
    /// Creates a lifetime group. On Windows, allocates a job object; elsewhere succeeds with no resources.
    pub fn new() -> io::Result<Self> {
        #[cfg(windows)]
        {
            let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if job == 0 || job == -1_isize as HANDLE {
                return Err(io::Error::last_os_error());
            }
            let basic = JOBOBJECT_BASIC_LIMIT_INFORMATION {
                LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                ..unsafe { std::mem::zeroed() }
            };
            let info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
                BasicLimitInformation: basic,
                ..unsafe { std::mem::zeroed() }
            };

            let r = unsafe {
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if r == 0 {
                unsafe {
                    CloseHandle(job);
                }
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                job: JobHandle(job),
            })
        }
        #[cfg(not(windows))]
        {
            Ok(Self {})
        }
    }

    /// Applies platform-specific options so the child exits when the bootstrapper dies (where supported).
    pub fn prepare_command(&self, cmd: &mut Command) {
        let _ = self;
        #[cfg(target_os = "linux")]
        linux::apply_parent_death_signal(cmd);
    }

    /// Registers a spawned direct child (required on Windows for job assignment).
    pub fn register_spawned(&self, child: &Child) -> io::Result<()> {
        #[cfg(windows)]
        {
            let raw = child.as_raw_handle();
            let ok = unsafe { AssignProcessToJobObject(self.job.0, raw as HANDLE) };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(not(windows))]
        {
            let _ = child;
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use std::io;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    /// Requests `SIGTERM` when the parent (bootstrapper) exits.
    pub(super) fn apply_parent_death_signal(cmd: &mut Command) {
        unsafe {
            cmd.pre_exec(|| {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_lifetime_group_new_succeeds() {
        let g = ChildLifetimeGroup::new();
        assert!(g.is_ok(), "{:?}", g.err());
    }
}
