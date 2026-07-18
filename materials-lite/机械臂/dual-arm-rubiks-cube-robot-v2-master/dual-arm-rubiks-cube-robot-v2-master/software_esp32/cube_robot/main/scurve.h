/**
 * @file scurve.h
 * @brief S-curve motion profile generator with jerk-limited trajectories
 */

 #ifndef SCURVE_SCURVE_H_
 #define SCURVE_SCURVE_H_
 
 #ifdef __cplusplus
 extern "C"
 {
 #endif
 
 #include <stddef.h>
 #include <stdint.h>
 
 /** Required buffer size for the S-curve generator object */
 // #define SCURVE_BUFFER_SIZE 392
 
 /**
  * @brief Motion profile generator states
  */
 typedef enum
 {
     CURVE_IDLE = 0,  /**< Ready for new profile */
     CURVE_INIT,      /**< Initialization phase */
     CURVE_CONF,      /**< Computing profile parameters */
     CURVE_BUSY,      /**< Executing profile segments */
     CURVE_ONEND,     /**< Profile complete */
 } scurve_state_e;
 
 /**
  * @brief Function return status codes
  */
 typedef enum
 {
     CURVE_SUCCESS = 0,        /**< Operation completed successfully */
     CURVE_INVALID_PARAMS,     /**< Invalid parameter values provided */
     CURVE_INVALID_STATE,      /**< Operation not allowed in current state */
 } scurve_return_status_e;
 
 typedef struct
 {
     float sign;   ///< Sign of the motion (1 or -1).
     float dist;   ///< Total distance to travel.
     float s1_end; // (seconds) cumulative end time of segment 1
     float s2_end; // segment 2 end, etc.
     float s3_end;
     float s4_end;
     float s5_end;
     float s6_end;
     float totalTime; // end of segment 7
 } scurve_param_t;
 
 typedef struct scurve_segment_state
 {
     float x0;                          // position at segment start
     float v0;                          // velocity at segment start
     float a0;                          // acceleration at segment start
     float j0;                          // jerk at segment start
     float t0;                          // time offset (when the segment started)
     float tEnd;                        // time when the segment ends
     int segmentIndex;                  // 1..7 or idle
     struct scurve_segment_state *next; // pointer to next segment state
 } scurve_segment_state_t;
 
 struct scurve_t_
 {
     scurve_state_e state; ///< Current motion profile state (IDLE, INIT, CONF, BUSY, QUICKSTOP).
     float Ts;             ///< Sample time (seconds).
     float pos_target;     ///< Target setpoint.
     float vel_start;      ///< Starting velocity.
     float vel_stop;       ///< Stopping velocity.
     float vel_max;        ///< Maximum allowed velocity in the profile.
     float acc_max;        ///< Maximum allowed acceleration in the profile.
     float dec_max;        ///< Maximum allowed deceleration in the profile.
     float jerk_max;       ///< Maximum allowed jerk in the profile.
     float vel_peak;       ///< Peak velocity in the profile.
     float vel_start_clip; ///< Start velocity after clipping.
     float pos_out;        ///< Profiled position output.
     float vel_out;        ///< Profiled velocity output.
     float acc_out;        ///< Profiled acceleration output.
     float jerk_out;       ///< Profiled jerk output.
 
     uint32_t profile_ticks; ///< Current tick counter for profile progression.
 
     // keep track of current segment
     int segStateIndex;
     scurve_segment_state_t segState[7];
 
     scurve_param_t param; ///< Precomputed parameters and thresholds for the motion profile.
 };
 
 /** Opaque type for the S-curve generator object */
 typedef struct scurve_t_ scurve_t;
 
 /**
  * Initializes an S-curve motion profile generator
  * 
  * @param obj Pointer to memory buffer for the scurve object
  * @param sampleTime_s Sample time in seconds for profile generation
  */
 void scurve_init(scurve_t *obj, float sampleTime_s);
 
 /**
  * @brief Execute one step of the motion profile generator
  * 
  * This function should be called periodically at the configured sample time.
  * It handles state transitions and trajectory generation.
  * 
  * @param obj Pointer to generator object
  * @return Current state of the generator
  */
 scurve_state_e scurve_run(scurve_t *obj);
 
 /**
  * @brief Start execution of a new motion profile
  * 
  * Validates parameters and transitions from IDLE to CONF state if:
  * - Current state is IDLE
  * - Maximum velocity > start and stop velocities
  * - All limits (velocity, acceleration, deceleration, jerk) are positive
  * 
  * @param obj Pointer to generator object
  * @return Status code indicating success or failure reason
  */
 scurve_return_status_e scurve_startProfile(scurve_t *obj);
 
 /**
  * @brief Get current state of the generator
  * 
  * @param obj Pointer to generator object
  * @return Current state
  */
 scurve_state_e scurve_getState(scurve_t *obj);
 
 /**
  * @brief Set sample time for profile generation
  * Only allowed in IDLE state
  * 
  * @param obj Pointer to generator object
  * @param sampleTime_s Sample time in seconds
  */
 void scurve_setSampleTime(scurve_t *obj, float sampleTime_s);
 
 /**
  * @brief Set target position for the motion profile
  * Only allowed in IDLE state
  * 
  * @param obj Pointer to generator object
  * @param pos_target Target position
  */
 void scurve_setPositionTarget(scurve_t *obj, float pos_target);
 
 /**
  * @brief Set current position output
  * Only allowed in IDLE state
  * 
  * @param obj Pointer to generator object
  * @param pos_out Current position
  */
 void scurve_setPositionOutput(scurve_t *obj, float pos_out);
 
 /**
  * @brief Set initial velocity for the motion profile
  * Only allowed in IDLE state
  * Must be less than maximum velocity
  * 
  * @param obj Pointer to generator object
  * @param vel_start Initial velocity
  */
 void scurve_setVelocityStart(scurve_t *obj, float vel_start);
 
 /**
  * @brief Set final velocity for the motion profile
  * Only allowed in IDLE state
  * Must be less than maximum velocity
  * 
  * @param obj Pointer to generator object
  * @param vel_stop Final velocity
  */
 void scurve_setVelocityStop(scurve_t *obj, float vel_stop);
 
 /**
  * @brief Set maximum allowed velocity
  * Only allowed in IDLE state
  * Must be greater than both start and stop velocities
  * 
  * @param obj Pointer to generator object
  * @param vel Maximum velocity
  */
 void scurve_setVelocityMax(scurve_t *obj, float vel);
 
 /**
  * @brief Set maximum allowed acceleration
  * Only allowed in IDLE state
  * Must be positive
  * 
  * @param obj Pointer to generator object
  * @param acc Maximum acceleration
  */
 void scurve_setAccelerationMax(scurve_t *obj, float acc);
 
 /**
  * @brief Set maximum allowed deceleration
  * Only allowed in IDLE state
  * Must be positive
  * 
  * @param obj Pointer to generator object
  * @param dec Maximum deceleration
  */
 void scurve_setDecelerationMax(scurve_t *obj, float dec);
 
 /**
  * @brief Set maximum allowed jerk
  * Only allowed in IDLE state
  * Must be positive
  * 
  * @param obj Pointer to generator object
  * @param jerk Maximum jerk
  */
 void scurve_setJerkMax(scurve_t *obj, float jerk);
 
 /**
  * @brief Get current motion profile outputs
  * 
  * @param obj Pointer to generator object
  * @param pos Position output
  * @param vel Velocity output
  * @param acc Acceleration output
  * @param jrk Jerk output
  */
 void scurve_getOutputs(scurve_t *obj, float *pos, float *vel, float *acc, float *jrk);
 
 /**
  * @brief Get current position output
  * 
  * @param obj Pointer to generator object
  * @return Current position
  */
 float scurve_getPositionOutput(scurve_t *obj);
 
 /**
  * @brief Get current velocity output
  * 
  * @param obj Pointer to generator object
  * @return Current velocity
  */
 float scurve_getVelocityOutput(scurve_t *obj);
 
 /**
  * @brief Get current acceleration output
  * 
  * @param obj Pointer to generator object
  * @return Current acceleration
  */
 float scurve_getAccelerationOutput(scurve_t *obj);
 
 /**
  * @brief Get current jerk output
  * 
  * @param obj Pointer to generator object
  * @return Current jerk
  */
 float scurve_getJerkOutput(scurve_t *obj);
 
 /**
  * @brief Get current profile tick counter
  * 
  * @param obj Pointer to generator object
  * @return Number of ticks since profile start
  */
 uint32_t scurve_getProfileTick(scurve_t *obj);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* SCURVE_SCURVE_H_ */
 