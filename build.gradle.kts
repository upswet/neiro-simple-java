plugins {
	java
	application
}

group = "neiro"
version = "0.0.1-SNAPSHOT"
description = "Neiro simple"

java {
	toolchain {
		languageVersion = JavaLanguageVersion.of(25)
	}
}

configurations {
	compileOnly {
		extendsFrom(configurations.annotationProcessor.get())
	}
}

repositories {
	mavenCentral()
}

dependencies {
	compileOnly("org.projectlombok:lombok:1.18.44")
	annotationProcessor("org.projectlombok:lombok:1.18.44")
	implementation("ch.qos.logback:logback-classic:1.5.18")

	testCompileOnly("org.projectlombok:lombok:1.18.44")
	testAnnotationProcessor("org.projectlombok:lombok:1.18.44")
}

application {
	mainClass.set("neiro.simple.ServerApplication")
}
